
import numbers
from typing import Collection, Any, Mapping
import numpy

from ..workbench.em_framework import parameters as workbench_param
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from ..util import distributions, DistributionTypeError, DistributionFreezeError
from ..util import make_rv_frozen, rv_frozen_as_dict
from .names import ShortnameMixin, TaggableMixin

def standardize_parameter_type(original_type):
    """Standardize parameter type descriptions

    Args:
        original_type (str): The original type

    Returns:
        str: The standarized type name
    """
    original_type = original_type.lower()
    if 'unc' in original_type:
        return 'uncertainty'
    if 'lev' in original_type:
        return 'lever'
    elif 'con' in original_type or 'fix' in original_type:
        return 'constant'
    raise ValueError('cannot decipher parameter ptype')

def standardize_data_type(original_type):
    dtype = str(original_type).lower()
    dtype = {
        'float': 'real',
        'float32': 'real',
        'float64': 'real',
        'floating': 'real',
        'double': 'real',
        'integer': 'int',
        'int32': 'int',
        'int64': 'int',
        'long': 'int',
        'boolean': 'bool',
        'category': 'cat',
        'categorical': 'cat',
    }.get(dtype, dtype)
    return dtype


def _get_bounds_from_dist(dist):
    ppf_zero = 0
    try:
        if isinstance(dist.dist, stats.rv_discrete):
            # ppf at actual zero for rv_discrete gives lower bound - 1
            # due to a quirk in the scipy.stats implementation
            # so we use the smallest positive float instead
            ppf_zero = 5e-324
    except AttributeError:
        pass
    lower_bound = dist.ppf(ppf_zero)
    upper_bound = dist.ppf(1.0)
    return lower_bound, upper_bound

def _get_lower_bound_from_dist(dist):
    ppf_zero = 0
    try:
        if isinstance(dist.dist, stats.rv_discrete):
            # ppf at actual zero for rv_discrete gives lower bound - 1
            # due to a quirk in the scipy.stats implementation
            # so we use the smallest positive float instead
            ppf_zero = 5e-324
    except AttributeError:
        pass
    lower_bound = dist.ppf(ppf_zero)
    return lower_bound

def _get_upper_bound_from_dist(dist):
    upper_bound = dist.ppf(1.0)
    return upper_bound

def make_parameter(
        name,
        ptype='constant',
        desc='missing description',
        min=None,
        max=None,
        dist=None,
        default=None,
        corr=None,
        address=None,
        dtype='infer',
        values=None,
        resolution=None,
        shortname=None,
        abbrev=None,
        meta=None,
):
    """
    Factory method to build a Parameter or Constant for a model.

    This function will create an object of the appropriate (sub)class
    for the parameter or constant.

    Args:
        name (str): A name for this parameter. The name must be a `str`
            and ideally a valid Python identifier (i.e., begins with
            a letter or underscore, contains only letters, numerals, and
            underscores).
        ptype (str, default 'constant'): The type for this parameter, one
            of {'constant', 'uncertainty', 'lever'}.
        min (numeric, optional): The minimum value for this parameter.
        max (numeric, optional): The maximum value for this parameter.
        dist (str or Mapping or rv_frozen, optional): A definition of a distribution
            to use for this parameter.  Can be specified just as the name of the distribution
            when that distribution is parameterized only by the min and max
            (e.g., 'uniform'). If the distribution requires other parameters,
            this argument should be a Mapping, with keys including 'name' for
            the name of the distribution, as well as giving one or more
            named distributional parameters as appropriate. Or, just pass
            a rv_frozen object directly (see scipy.stats).  The distribution
            defined here is of primary use for uncertainty parameters, as the
            features of defined uncertainty distributions can be used to
            derive probability distributions on outputs.  However, distributions
            can also be used for policy lever parameters to guide the development
            of appropriate experimental designs.
        default (Any, optional): A default value for this parameter. The default
            value is used as the actual value for constant parameters. It is also
            used during univariate sensitivity testing as the value for this
            parameter when other parameters are being evaluated at non-default
            values.
        corr (dict, optional): A correlation definition that relates this parameter
            to others. Only applicable for uncertainty parameters.
        address (Any, optional): The address to use to access this parameter in
            the model.  This is an implementation-specific detail. For example,
            in an Excel-based model, the address could be a sheet and cell reference
            given as a string.
        dtype (str, default 'infer'): A dtype for this parameter, one
            of {'cat', 'int', 'real', 'bool'} or some sub-class variant or specialization
            thereof (e.g., int64).
        values (Collection, optional): A collection of possible values, relevant only
            for categorical parameters.
        resolution (Collection, optional): A collection of possible particular values,
            used to set the possible values considered when sampling with factorial-based
            designs.
        shortname (str, optional):
            A shorter name, especially useful when the name of this parameter
            is a long strings that may not display neatly in figures.
        abbrev (Mapping, optional):
            A set of abbreviations used for values, especially useful when the names of
            values are long strings that may not display neatly in figures.
        meta (Any, optional):
            Meta-data to attach to this parameter

    Returns:
        Parameter or Constant
    """

    # Convert dist to a Mapping if it is just a string
    if isinstance(dist, str):
        dist = {'name': dist}

    # Default correlation is an empty list.
    corr = corr if corr is not None else []

    # Standardize the dtype to a lowercase string of
    # correct type
    dtype = standardize_data_type(dtype)

    if dtype == 'infer':
        if values is not None:
            if set(values) == {True, False}:
                dtype = 'bool'
            else:
                dtype = 'cat'
        elif max is True and min is False:
            dtype = 'bool'
        elif isinstance(min, numbers.Integral) and isinstance(max, numbers.Integral):
            dtype = 'int'
        elif isinstance(min, numbers.Real) and isinstance(max, numbers.Real):
            dtype = 'real'
        elif dist is not None and isinstance(dist, Mapping):
            try:
                make_rv_frozen(**dist, discrete=True)
            except DistributionTypeError:
                dtype = 'real'
            else:
                dtype = 'int'
        else:
            raise ValueError(f'cannot infer dtype for {name}, give it explicitly')

    if dtype not in ('cat', 'int', 'real', 'bool'):
        message = f"invalid dtype {dtype} for parameter {name}, must be 'cat', 'int', 'real', or 'bool'"
        if 'unc' in dtype:
            message += "\n(to set this parameter as an uncertainty, set `ptype` to 'uncertainty', not `dtype`)"
        elif 'lev' in dtype:
            message += "\n(to set this parameter as a lever, set `ptype` to 'lever', not `dtype`)"
        elif 'con' in dtype:
            message += "\n(to set this parameter as a constant, set `ptype` to 'constant', not `dtype`)"
        raise ValueError(message)

    # Data checks

    if dist is not None and not isinstance(dist, Mapping) and not isinstance(dist, rv_frozen):
        raise TypeError(f'dist must be a dict or rv_frozen for {name}, not {type(dist)}')

    if dist is None:
        dist_ = {}
        rv_gen = None
    elif isinstance(dist, rv_frozen):
        dist_ = {'name': dist.dist.name}
        dist_.update(dist.kwds)
        rv_gen = dist
    else:
        dist_ = dist
        rv_gen = None

    dist_for_maker = dist_.copy()
    if min is not None:
        dist_for_maker['min'] = min
    if max is not None:
        dist_for_maker['max'] = max

    if dtype=='bool':
        dist_for_maker['min'] = 0
        dist_for_maker['max'] = 1
    elif dtype=='cat':
        dist_for_maker['min'] = 0
        dist_for_maker['max'] = (len(values)-1) if values is not None else 0


    ptype = standardize_parameter_type(ptype)

    if ptype is 'constant':
        if dist_.get('name') is None:
            dist_['name'] = 'constant'
        if dist_.get('name') != 'constant':
            raise ValueError(f'constant cannot have non-constant distribution for {name}')
        rv_gen_tentative = None
    else:
        # If inferred dtype is int but distribution is discrete, promote to real
        try:
            rv_gen_tentative = rv_gen or make_rv_frozen(**dist_for_maker, discrete=True)
        except DistributionTypeError:
            if dtype == 'int':
                dtype = 'real'
            rv_gen_tentative = rv_gen or make_rv_frozen(**dist_for_maker, discrete=False)
        except DistributionFreezeError as err:
            raise DistributionFreezeError(f'on freeze for {name}') from err



    if dtype == 'bool':
        if min is None:
            min = False
        if min != False:
            raise ValueError(f'min of bool must be False for {name}')
        if max is None:
            max = True
        if max != True:
            raise ValueError(f'max of bool must be True for {name}')
        values = [False, True]

    if dtype in {'int', 'real'}:
        if dist_.get('name') == 'constant':
            if min is None and default is not None:
                min = default
            if max is None and default is not None:
                max = default

        if rv_gen_tentative is not None:
            min, max = _get_bounds_from_dist(rv_gen_tentative)

        if min is None:
            raise ValueError(f'min of {dtype} is required for {name}')
        if max is None:
            raise ValueError(f'max of {dtype} is required for {name}')

    # Create the Parameter
    if ptype == 'constant':
        if dtype == 'cat':
            p = Constant(name, default, desc=desc, address=address, dtype='cat', meta=meta)
        else:
            p = Constant(name, default, desc=desc, address=address, meta=meta)
    elif dtype == 'cat':
        p = CategoricalParameter(
            name,
            values,
            default=default,
            desc=desc,
            address=address,
            ptype=ptype,
            corr=corr,
            abbrev=abbrev,
            shortname=shortname,
            meta=meta,
        )
    elif dtype == 'int':
        rv_gen = rv_gen or make_rv_frozen(**dist_for_maker, discrete=True)

        if rv_gen is None:
            raise ValueError(f'failed to make {name} ({ptype}) from {dist_}')
            # p = Constant(name, default, desc=desc, address=address)
        else:
            p = IntegerParameter(
                name,
                lower_bound=min,
                upper_bound=max,
                resolution=resolution,
                default=default,
                dist=rv_gen,
                dist_def=dist_,
                desc=desc,
                address=address,
                ptype=ptype,
                corr=corr,
                abbrev=abbrev,
                shortname=shortname,
                meta=meta,
            )
    elif dtype == 'real':
        rv_gen = rv_gen or make_rv_frozen(**dist_for_maker)

        if rv_gen is None:
            raise ValueError(f'failed to make {name} ({ptype}) from {dist_}')
            # p = Constant(name, default, desc=desc, address=address)
        else:
            p = RealParameter(
                name,
                lower_bound=min,
                upper_bound=max,
                resolution=resolution,
                default=default,
                dist=rv_gen,
                dist_def=dist_,
                desc=desc,
                address=address,
                ptype=ptype,
                corr=corr,
                abbrev=abbrev,
                shortname=shortname,
                meta=meta,
            )

    elif dtype == 'bool':
        rv_gen = rv_gen or make_rv_frozen(**dist_for_maker, discrete=True)
        if rv_gen is None:
            raise ValueError(f'failed to make {name} ({ptype}) from {dist_}')
            # p = Constant(name, default, desc=desc, address=address)
        else:
            p = BooleanParameter(
                name,
                default=default,
                dist=rv_gen,
                dist_def=dist_,
                desc=desc,
                address=address,
                ptype=ptype,
                corr=corr,
                abbrev=abbrev,
                shortname=shortname,
                meta=meta,
            )
    else:
        raise ValueError(f"invalid dtype {dtype}")

    return p



class Constant(workbench_param.Constant):

    ptype = 'constant'
    """str: Parameter type, for compatibility with Parameter."""

    def __init__(self, name, value, desc="", address=None, dtype=None, meta=None):

        if value is None:
            raise ValueError("Constant.value cannot be None")

        workbench_param.Constant.__init__(
            self,
            name,
            value,
        )

        self.desc = desc
        """str: Human readable description of this constant, for reference only"""

        self.address = address
        """
        Any: The address to use to access this parameter in the model.

        This is an implementation-specific detail. For example,
        in an Excel-based model, the address could be a sheet and cell reference
        given as a string.
        """

        if dtype is None:
            dtype = numpy.asarray(value).dtype

        dtype = standardize_data_type(dtype)

        self.dtype = dtype
        """str: The dtype for the value, as a string."""

        self.meta = meta
        """Any: meta-data that is attached to this constant."""

    @property
    def default(self):
        """Read-only alias for value"""
        return self.value

    @property
    def values(self):
        """list: The value as a one-item list"""
        return [self.value,]

    def __eq__(self, other):
        try:
            if self.address != other.address:
                return False
            if self.dtype != other.dtype:
                return False
        except AttributeError:
            return False
        return super().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

class Parameter(workbench_param.Parameter, ShortnameMixin, TaggableMixin):

    dtype = None

    def __init__(
            self,
            name,
            dist,
            *,
            lower_bound=None,
            upper_bound=None,
            resolution=None,
            default=None,
            variable_name=None,
            pff=False,
            desc="",
            address=None,
            ptype=None,
            corr=None,
            dist_def=None,
            shortname=None,
            abbrev=None,
            tags=None,
            meta=None,
    ):

        # The default constructor for ema_workbench parameters uses no distribution
        # But for EMAT, we want to always define a distribution explicitly
        # for clarity.

        if dist is None and (lower_bound is None or upper_bound is None):
            raise ValueError("must give lower_bound and upper_bound, or dist")

        if dist is None:
            from scipy.stats import uniform
            dist = uniform(lower_bound, upper_bound-lower_bound)

        if isinstance(dist, str):
            dist = {'name':dist}

        if isinstance(dist, Mapping):
            dist = dict(**dist)
            if lower_bound is not None:
                dist['min'] = lower_bound
            if upper_bound is not None:
                dist['max'] = upper_bound
            dist = make_rv_frozen(**dist)

        # We extract and set the lower and upper bounds here,
        # in order to use the default constructor from the workbench.

        ppf_zero = 0
        if isinstance(dist.dist, stats.rv_discrete):  # @UndefinedVariable
            # ppf at actual zero for rv_discrete gives lower bound - 1
            # due to a quirk in the scipy.stats implementation
            # so we use the smallest positive float instead
            ppf_zero = 5e-324

        lower_bound = dist.ppf(ppf_zero)
        upper_bound = dist.ppf(1.0)

        if self.dtype == 'int':
            lower_bound = int(lower_bound)
            upper_bound = int(upper_bound)

        workbench_param.Parameter.__init__(
            self,
            name=name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            resolution=resolution,
            default=default,
            variable_name=variable_name,
            pff=pff,
        )

        self.dist = dist

        self.desc = desc
        """str: Human readable description of this parameter, for reference only"""

        self.address = address
        """
        Any: The address to use to access this parameter in the model.

        This is an implementation-specific detail. For example,
        in an Excel-based model, the address could be a sheet and cell reference
        given as a string.
        """

        self.ptype = standardize_parameter_type(ptype) if ptype is not None else None
        """str: Parameter type, one of {'constant', 'uncertainty', 'lever'}"""

        self.corr = corr if corr is not None else []
        """Dict: A correlation definition.  Key give names of other parameters, values give correlation."""

        self.dist_def = dict(dist_def) if dist_def is not None else {}
        """Dict: The arguments that define the underlying distribution."""

        self._shortname = shortname

        self.abbrev = abbrev or {}
        """Dict: Abbreviations used for long attribute names in figures."""

        if tags:
            if isinstance(tags, str):
                tags = [tags]
            for tag in tags:
                self.add_tag(tag)

        self.meta = meta
        """Any: meta-data that is attached to this object."""

    @property
    def min(self):
        return self.lower_bound

    @property
    def max(self):
        return self.upper_bound

    @property
    def range(self):
        return self.lower_bound, self.upper_bound

    def __eq__(self, other):
        try:
            if type(self) != type(other):
                return False
            if self.address != other.address:
                return False
            if self.dtype != other.dtype:
                return False
            if self.ptype != other.ptype:
                return False
            if self.corr != other.corr:
                return False
            if self.distdef != other.distdef:
                return False
        except AttributeError:
            return False
        if not isinstance(self, other.__class__):
            return False

        self_keys = set(self.__dict__.keys())
        other_keys = set(other.__dict__.keys())
        if self_keys - other_keys:
            return False
        else:
            for key in self_keys:
                if key == 'dist_def':
                    continue
                if key != 'dist':
                    if getattr(self, key) != getattr(other, key):
                        return False
                else:
                    # name, parameters
                    self_dist = getattr(self, key)
                    other_dist = getattr(other, key)
                    if self_dist.dist.name != other_dist.dist.name:
                        return False
                    if self_dist.args != other_dist.args:
                        return False

            else:
                return True

    def explain_neq(self, other):
        try:
            if type(self) != type(other):
                return f"not same type: {type(self)} != {type(other)}"
            if self.address != other.address:
                return f"not same address: {self.address} != {other.address}"
            if self.dtype != other.dtype:
                return f"not same dtype: {self.dtype} != {other.dtype}"
            if self.ptype != other.ptype:
                return f"not same ptype: {self.ptype} != {other.ptype}"
            if self.corr != other.corr:
                return f"not same corr: {self.corr} != {other.corr}"
            if self.distdef != other.distdef:
                return f"not same distdef: {self.distdef} != {other.distdef}"
        except AttributeError:
            return f"key attributes not all defined"
        if not isinstance(self, other.__class__):
            return f"not nested classes {self.__class__} is not {other.__class__}"
        self_keys = set(self.__dict__.keys())
        other_keys = set(other.__dict__.keys())
        if self_keys - other_keys:
            return f"misaligned keys {self_keys - other_keys}"
        else:
            for key in self_keys:
                if key == 'dist_def':
                    continue
                if key != 'dist':
                    if getattr(self, key) != getattr(other, key):
                        return f"not same attr {key}: {getattr(self, key)} != {getattr(other, key)}"
                else:
                    # name, parameters
                    self_dist = getattr(self, key)
                    other_dist = getattr(other, key)
                    if self_dist.dist.name != other_dist.dist.name:
                        return f"not same dist.name: {self_dist.dist.name} != {other_dist.dist.name}"
                    if self_dist.args != other_dist.args:
                        return f"not same dist.args: {self_dist.args} != {other_dist.args}"
            else:
                return None

    @property
    def distdef(self):
        result = rv_frozen_as_dict(self.dist, self.min, self.max)
        return result

    @property
    def dist_description(self):
        result = self.dist_def
        if isinstance(result, str):
            result = {'name':result}
        if result.get('name') == 'pert':
            if 'gamma' in result:
                return f"Pert Distribution (min={self.min}, peak={result.get('peak')}, max={self.max}, gamma={result.get('gamma')})"
            return f"Pert Distribution (min={self.min}, peak={result.get('peak')}, max={self.max})"
        if result.get('name') == 'uniform' or len(result)==0:
            return f"Uniform Distribution (min={self.min}, max={self.max})"
        if result.get('name') == 'triangle':
            return f"Trianglar Distribution (min={self.min}, peak={result.get('peak')}, max={self.max})"
        return str(result)

    def __repr__(self):
        return f"<emat.{self.__class__.__name__} '{self.name}'>"

    def _hash_it(self, ha=None):
        from ..util.hasher import hash_it
        return hash_it(
            self.name,
            self.dist_def,
            self.resolution,
            self.address,
            self.dtype,
            self.pff,
            tuple(self.variable_name),
            self.shortname,
            self.ptype,
            self.corr,
            ha=ha,
        )

    def get_abbrev(self, name):
        """Get an abbreviated name if available."""
        try:
            return self.abbrev.get(name, name)
        except AttributeError:
            return name

    def set_abbrev(self, values=None, **kwargs):
        if values is None:
            values = {}
        try:
            self.abbrev.update(values, **kwargs)
        except AttributeError:
            self.abbrev = {}
            self.abbrev.update(values, **kwargs)

class RealParameter(Parameter, workbench_param.RealParameter):

    dtype = 'real'

    def __init__(self, name, *, lower_bound=None, upper_bound=None, resolution=None,
                 default=None, variable_name=None, pff=False, dist=None, dist_def=None,
                 desc="", address=None, ptype=None, corr=None, shortname=None, abbrev=None,
                 meta=None):

        if dist is None and (lower_bound is None or upper_bound is None):
            raise ValueError("must give lower_bound and upper_bound, or dist")

        if dist is None:
            from scipy.stats import uniform
            dist = uniform(lower_bound, upper_bound-lower_bound)

        Parameter.__init__(
            self,
            name,
            dist=dist,
            resolution=resolution,
            default=default,
            variable_name=variable_name,
            pff=pff,
            desc=desc,
            address=address,
            ptype=ptype,
            corr=corr,
            dist_def=dist_def,
            shortname=shortname,
            abbrev=abbrev,
            meta=meta,
        )
        

    @property
    def min(self):
        return float(super().lower_bound)

    @property
    def max(self):
        return float(super().upper_bound)


class IntegerParameter(Parameter, workbench_param.IntegerParameter):

    dtype = 'int'

    def __init__(self, name, *, lower_bound=None, upper_bound=None, resolution=None,
                 default=None, variable_name=None, pff=False, dist=None, dist_def=None,
                 desc="", address=None, ptype=None, corr=None, shortname=None, abbrev=None,
                 meta=None):

        if dist is None and (lower_bound is None or upper_bound is None):
            raise ValueError("must give lower_bound and upper_bound, or dist")        

        if dist is None:
            from scipy.stats import randint
            dist = randint(lower_bound, upper_bound+1)

        Parameter.__init__(
            self,
            name,
            dist=dist,
            resolution=resolution,
            default=default, variable_name=variable_name, pff=pff,
            desc=desc, address=address, ptype=ptype, corr=corr,
            dist_def=dist_def,
            shortname=shortname,
            abbrev=abbrev,
            meta=meta,
        )

        if self.resolution is not None:
            for entry in self.resolution:
                if not isinstance(entry, numbers.Integral):
                    raise ValueError(('all entries in resolution should be '
                                      'integers'))

    @property
    def min(self):
        return int(super().lower_bound)

    @property
    def max(self):
        return int(super().upper_bound)



class BooleanParameter(Parameter, workbench_param.BooleanParameter):

    dtype = 'bool'

    def __init__(self, name, *, lower_bound=None, upper_bound=None, resolution=None,
                 default=None, variable_name=None, pff=False, dist=None, dist_def=None,
                 desc="", address=None, ptype=None, corr=None, shortname=None, abbrev=None,
                 meta=None):

        Parameter.__init__(
            self,
            name,
            dist=dist,
            resolution=resolution,
            default=default, variable_name=variable_name, pff=pff,
            desc=desc, address=address, ptype=ptype, corr=corr,
            dist_def=dist_def,
            shortname=shortname,
            abbrev=abbrev,
            meta=meta,
        )

        cats = [workbench_param.create_category(cat) for cat in [False, True]]

        self._categories = workbench_param.NamedObjectMap(workbench_param.Category)

        self.categories = cats
        self.resolution = [i for i in range(len(self.categories))]
        self.multivalue = False


    @property
    def values(self):
        """List: The possible discrete values."""
        return [False, True]

    @property
    def min(self):
        return False

    @property
    def max(self):
        return True


class CategoricalParameter(Parameter, workbench_param.CategoricalParameter):

    dtype = 'cat'

    def __init__(self, name, categories, *, default=None, variable_name=None,
                 pff=False, multivalue=False,
                 desc="", address=None, ptype=None, corr=None,
                 dist=None, singleton_ok=False, shortname=None, abbrev=None,
                 meta=None):
        lower_bound = 0
        upper_bound = len(categories) - 1

        from scipy.stats import randint
        dist = randint(lower_bound, upper_bound+1)

        if upper_bound == 0 and not singleton_ok:
            raise ValueError('there should be more than 1 category')

        Parameter.__init__(
            self,
            name,
            dist=dist,
            resolution=None,
            default=default, variable_name=variable_name, pff=pff,
            desc=desc, address=address, ptype=ptype, corr=corr,
            shortname=shortname,
            abbrev=abbrev,
            meta=meta,
        )

        cats = [workbench_param.create_category(cat) for cat in categories]

        self._categories = workbench_param.NamedObjectMap(workbench_param.Category)

        self.categories = cats
        self.resolution = [i for i in range(len(self.categories))]
        self.multivalue = multivalue

    @property
    def values(self):
        """List: The possible discrete values."""
        return list(i.value for i in self.categories)

    @property
    def min(self):
        """None: Categorical parameters are not characterized by a lower bound."""
        return None

    @property
    def max(self):
        """None: Categorical parameters are not characterized by an upper bound."""
        return None

    @property
    def distdef(self):
        """None: Categorical parameters distribution is not implemented."""
        return None

    @property
    def dist_description(self):
        return "Uniform Distribution"

#############

class OLD_Parameter:
    """
    Definitions for a particular input for a model.

    Args:
        name (str): A name for this parameter. The name must be a `str`
            and ideally a valid Python identifier (i.e., begins with
            a letter or underscore, contains only letters, numerals, and
            underscores).
        ptype (str, default 'constant'): The type for this parameter, one
            of {'constant', 'uncertainty', 'lever'}.
        min (numeric, optional): The minimum value for this parameter.
        max (numeric, optional): The maximum value for this parameter.
        dist (str or Mapping, optional): A definition of a distribution
            to use for this parameter, which is only relevant for uncertainty
            parameters.  Can be specified just as the name of the distribution
            when that distribution is parameterized only by the min and max
            (e.g., 'uniform'). If the distribution requires other parameters,
            this argument should be a Mapping, with keys including 'name' for
            the name of the distribution, as well as giving one or more
            named distributional parameters as appropriate.
        default (Any, optional): A default value for this parameter. The default
            value is used as the actual value for constant parameters. It is also
            used during univariate sensitivity testing as the value for this
            parameter when other parameters are being evaluated at non-default
            values.
        corr (dict, optional): A correlation definition that relates this parameter
            to others. Only applicable for uncertainty parameters.
        address (Any, optional): The address to use to access this parameter in
            the model.  This is an implementation-specific detail. For example,
            in an Excel-based model, the address could be a sheet and cell reference
            given as a string.
        dtype (str, default 'infer'): A dtype for this parameter, one
            of {'cat', 'int', 'real', 'bool'} or some sub-class variant or specialization
            thereof (e.g., int64).
        values (Collection, optional): A collection of possible values, relevant only
            for categorical parameters.
        resolution (Collection, optional): A collection of possible particular values,
            used to set the possible values considered when sampling with factorial-based
            designs.

    """

    def __init__(
            self,
            name,
            ptype='constant',
            desc='missing description',
            min=None,
            max=None,
            dist=None,
            default=None,
            corr=None,
            address=None,
            dtype='infer',
            values=None,
            resolution=None,
    ):
        self.name = name
        """str: Parameter name, used to identify parameter."""

        self.ptype = standardize_parameter_type(ptype)
        """str: Parameter type, one of {'constant', 'uncertainty', 'lever'}"""

        self.desc = desc
        """str: Human readable description of this parameter, for reference only"""

        self.min = min
        """numeric: Lower bound for this parameter, or None"""

        self.max = max
        """numeric: Upper bound for this parameter, or None"""

        if isinstance(dist, str):
            dist = {'name': dist}
        self.dist = dist
        """Dict: definition of a distribution to use for this parameter"""

        self.default = default
        """A default value for this parameter, used for constants or in univariate sensitivity testing"""

        self.corr = corr if corr is not None else []
        self.address = address
        self.dtype = standardize_data_type(dtype)
        self.values = values
        self.resolution = resolution

        if self.dtype == 'infer':
            if self.values is not None:
                if set(self.values) == {True, False}:
                    self.dtype = 'bool'
                else:
                    self.dtype = 'cat'
            elif self.max is True:
                self.dtype = 'bool'
            elif isinstance(self.max, numbers.Integral):
                self.dtype = 'int'
            elif isinstance(self.max, numbers.Real):
                self.dtype = 'real'
            else:
                self.dtype = 'bool'

        if self.dtype not in ('cat','int','real','bool'):
            raise ValueError(f"invalid dtype {self.dtype}")

        # Data checks

        if self.dist is not None and not isinstance(self.dist, Mapping):
            raise TypeError(f'dist must be a dict for {self.name}, not {type(self.dist)}')

        if self.ptype is 'constant':
            if self.dist is None:
                self.dist = {'name': 'constant'}
            if self.dist.get('name') != 'constant':
                raise ValueError(f'constant cannot have non-constant distribution for {self.name}')

        if self.dtype == 'bool':
            if self.min is None:
                self.min = False
            if self.min != False:
                raise ValueError(f'min of bool must be False for {self.name}')
            if self.max is None:
                self.max = True
            if self.max != True:
                raise ValueError(f'max of bool must be True for {self.name}')
            self.values = [False, True]

        if self.dtype in {'int','real'}:
            if self.dist is not None and self.dist.get('name') == 'constant':
                if self.min is None and self.default is not None:
                    self.min = self.default
                if self.max is None and self.default is not None:
                    self.max = self.default

            if self.min is None:
                raise ValueError(f'min of {self.dtype} is required for {self.name}')
            if self.max is None:
                raise ValueError(f'max of {self.dtype} is required for {self.name}')


    def get_parameter(self):
        """Get an ema_workbench.Parameter from this emat.Parameter.

        This method returns an
        :class:`ema_workbench.Parameter <ema_workbench.em_framework.parameters.Parameter>`
        of the correct type for this :class:`emat.Parameter`.  This will be one of:

        * :class:`Constant <ema_workbench.em_framework.parameters.Constant>`
        * :class:`CategoricalParameter <ema_workbench.em_framework.parameters.CategoricalParameter>`
        * :class:`IntegerParameter <ema_workbench.em_framework.parameters.IntegerParameter>`
        * :class:`RealParameter <ema_workbench.em_framework.parameters.RealParameter>`
        * :class:`BooleanParameter <ema_workbench.em_framework.parameters.BooleanParameter>`

        """
        if self.ptype == 'constant':
            return Constant(self.name, self.default)
        elif self.dtype == 'cat':
            return CategoricalParameter(self.name, self.values, default=self.default,)
        elif self.dtype == 'int':
            return IntegerParameter(
                self.name, self.min, self.max, resolution=self.resolution,
                default=self.default,
            )
        elif self.dtype == 'real':
            if self.dist is not None and len(self.dist) > 0:

                _d = self.dist.copy()
                distname = _d.pop('name', 'uniform')

                if distname == 'uniform':
                    _d['loc'] = self.min
                    _d['scale'] = self.max - self.min
                elif distname == 'triangle':
                    _d['lower_bound'] = self.min
                    _d['upper_bound'] = self.max
                elif distname == 'pert':
                    _d['lower_bound'] = self.min
                    _d['upper_bound'] = self.max

                dist_gen = getattr(distributions, distname)
                dist = dist_gen(**_d)

            else:
                dist = None
            return RealParameter(
                self.name, self.min, self.max, resolution=self.resolution,
                default=self.default,
                dist=dist,
            )
        elif self.dtype == 'bool':
            return BooleanParameter(self.name, default=self.default)
        else:
            raise ValueError(f"invalid dtype {self.dtype}")

    def __repr__(self):
        classname = self.__class__.__name__
        name = self.name
        rep = f"{classname}('{name}', dtype={self.dtype}, ptype='{self.ptype}'"
        rep += ')'
        return rep

    def __eq__(self, other):
        keys = {'name', 'ptype', 'desc', 'dtype', 'default',
                'min', 'max', 'dist', 'corr', 'values',
                'address', 'resolution'}
        for k in keys:
            if getattr(self, k) != getattr(other, k):
                return False
        return True
