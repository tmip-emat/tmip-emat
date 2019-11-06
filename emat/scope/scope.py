# -*- coding: utf-8 -*-

import pandas
import numpy
import yaml
import warnings
import itertools
from ema_workbench import ScalarOutcome
from ema_workbench.em_framework.parameters import Category
from typing import Mapping
from scipy.stats._distn_infrastructure import rv_frozen

from ..database.database import Database
from .parameter import Parameter, standardize_parameter_type, make_parameter
from .measure import Measure
from ..util.docstrings import copydoc
from ..util import rv_frozen_as_dict

from ..exceptions import *

def _name_or_dict(x):
    if isinstance(x, rv_frozen):
        x = rv_frozen_as_dict(x)
    if x is None or not isinstance(x, Mapping):
        return x
    if len(x)>1:
        return x
    if list(x.keys()) == ['name']:
        return x['name']
    return x

def _as_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except:
        return x

class Scope:
    '''Definitions for the relevant inputs and outputs for a model.

    A Scope provides a structure to define the nature of the inputs
    and outputs for exploratory modeling.

    Args:
        scope_file (str): path to scope file
        scope_def (str, optional): The content of the scope file, if it has
            already been read into a string.  If this value is given, it is
            assumed to be the contents of the file and the file is not actually
            read again.
    '''

    scope_file = ''
    name = ''
    random_seed = 1234
    desc = ''
    xl_di = {}
    m_di = {}

    def __init__(self, scope_file, scope_def=None):

        self.scope_file = scope_file

        self._m_list = []
        """list of Measure: A list of performance measures that are output by the model."""

        self._x_list = []
        self._l_list = []
        self._c_list = []


        self.__parse_scope(scope_def=scope_def)
         
    def __parse_scope(self, scope_def=None):
        '''parser to read scope yaml file'''
        if scope_def is None:
            with open(self.scope_file, 'r') as stream:
                scope = yaml.load(stream, Loader=yaml.FullLoader)
        else:
            scope = yaml.load(scope_def, Loader=yaml.FullLoader)

        for k in ('scope', 'inputs', 'outputs'):
            if k not in scope:
                raise ScopeFormatError(f'scope file must include "{k}" as a top level key')

        self.name = str(scope['scope']['name'])
        self.desc = scope['scope'].get('desc', '')
        self.xl_di = scope['inputs']
        self.m_di = scope['outputs']

        if not isinstance(self.xl_di, dict):
            raise ScopeFormatError(
                'inputs must be a dictionary with (name: attributes) key:value pairs'
            )

        if not isinstance(self.m_di, dict):
            raise ScopeFormatError(
                'outputs must be a dictionary with (name: attributes) key:value pairs'
            )

        for m_name, m_attr in self.m_di.items():
            if isinstance(m_attr, dict):
                self._m_list.append(Measure(m_name, **m_attr))
            else:
                warnings.warn(f'for {m_name} cannot process list {m_attr}')
                self._m_list.append(Measure(m_name))

        if 'random_seed' in scope:
            self.random_seed = scope['random_seed']

        for x_name, x_attr in self.xl_di.items():
            if not isinstance(x_attr, dict):
                warnings.warn(f'for {x_name} cannot process list {x_attr}')
            else:
                x_attr_type = x_attr.get('ptype', 'missing')
                if x_attr_type == 'missing':
                    raise ScopeFormatError(f'inputs:{x_name} is missing ptype, must be uncertainty, lever, or constant')
                if not isinstance(x_attr_type, str):
                    raise ScopeFormatError(f'inputs:{x_name} has invalid ptype {x_attr_type}')
                try:
                    x_attr_type = standardize_parameter_type(x_attr_type)
                except ValueError:
                    raise ScopeFormatError(f'inputs:{x_name} has invalid ptype {x_attr.get("ptype")}')

                try:
                    p = make_parameter(x_name, **x_attr)
                except Exception as err:
                    raise ScopeFormatError(str(err))
                else:
                    if x_attr_type == 'uncertainty':
                        self._x_list.append(p)
                    elif x_attr_type == 'lever':
                        self._l_list.append(p)
                    elif x_attr_type == 'constant':
                        self._c_list.append(p)
                    else:
                        raise ScopeFormatError(f'inputs:{x_name} has invalid ptype {x_attr.get("ptype")}')


    def __eq__(self, other):
        if type(other) != type(self):
            return False
        for k in ('_x_list', '_l_list', '_c_list', ):
            if len(getattr(self,k)) != len(getattr(other,k)):
                return False
            for i,j in zip(getattr(self,k), getattr(other,k)):
                if isinstance(i, rv_frozen):
                    if rv_frozen_as_dict(i) != rv_frozen_as_dict(j):
                        return False
                else:
                    if i != j:
                        return False
        for k in ('_m_list', 'name', 'desc'):
            if getattr(self,k) != getattr(other,k):
                return False
        return True
    

    def store_scope(self, db: Database):
        ''' writes variables and scope definition to database 
        
            Required prior to running experiment design
            
            Not necessary if scope has already been run
            
            Args:
                db (Database): database object
        '''
                
        # load experiment variables and performance measures        
        db.init_xlm([(xl, self.xl_di[xl]['ptype']) for xl in self.xl_di],
                    [(m.name, m.transform) for m in self._m_list])
        
        # load scope definitions
        db.write_scope(self.name,
                      self.scope_file,
                      [xl for xl in self.xl_di],
                      [m.name for m in self._m_list],
                       content=self)
        
    def delete_scope(self, db: Database):
        '''Deletes scope from database.

        Args:
            db (Database): The database from which to delete this Scope.

        Note:
            Only the `name` attribute is used to identify the scope
            to delete.  If some other different scope is stored in
            the database with the same name as this scope, it will
            be deleted.
        '''
        db.delete_scope(self.name)

    def n_factors(self):
        '''Number of input factors defined in this scope.'''
        return len(self._c_list) + len(self._x_list) + len(self._l_list)

    def n_sample_factors(self):
        '''Number of non-constant input factors defined in this scope.'''
        return len(self._x_list) + len(self._l_list)

    @property
    def xl_list(self):
        return self._x_list + self._l_list

    @property
    def xlc_list(self):
        return self._x_list + self._l_list + self._c_list

    def __repr__(self):
        content = []
        if len(self._c_list):
            content.append(f"{len(self._c_list)} constants")
        if len(self._x_list):
            content.append(f"{len(self._x_list)} uncertainties")
        if len(self._l_list):
            content.append(f"{len(self._l_list)} levers")
        if len(self._m_list):
            content.append(f"{len(self._m_list)} measures")
        return f"<emat.Scope with " + ", ".join(content) + ">"


    def duplicate(
            self,
            strip_measure_transforms=False,
            include_measures=None,
            exclude_measures=None,
    ):
        """Create a duplicate scope, optionally stripping some features.

        Args:
            strip_measure_transforms (bool, default False):
                Remove the 'transform' values from all measures.
            include_measures (Collection[str], optional): If provided, only
                output performance measures with names in this set will be included.
            exclude_measures (Collection[str], optional): If provided, only
                output performance measures with names not in this set will be included.

        Returns:
            Scope
        """
        y = self.dump(strip_measure_transforms=strip_measure_transforms,
                      include_measures=include_measures,
                      exclude_measures=exclude_measures,)
        try:
            return type(self)(self.scope_file, scope_def=y)
        except:
            print("~"*20)
            print(y)
            print("~"*20)
            raise

    def dump(
            self,
            stream=None,
            filename=None,
            strip_measure_transforms=False,
            include_measures=None,
            exclude_measures=None,
            default_flow_style=False,
            **kwargs,
    ):
        """
        Serialize this scope into a YAML stream.

        Args:
            stream (file-like or None): Serialize into this stream. If None,
                return the produced string instead, unless `filename` is given.
            filename (path-like or None): If given and `stream` is None,
                then write the serialized result into this file.
            strip_measure_transforms (bool, default False): Remove the
                'transform' values from all measures in the output.
            include_measures (Collection[str], optional): If provided, only
                output performance measures with names in this set will be included.
            exclude_measures (Collection[str], optional): If provided, only
                output performance measures with names not in this set will be included.
            default_flow_style (bool, default False): Use the default_flow_style,
                see yaml.dump for details.
            **kwargs:
                All other keyword arguments are forwarded as-is to `yaml.dump`

        Returns:
            str:
                If both `stream` and `filename` are None, the serialized YAML
                content is returned as a string.

        Raises:
            FileExistsError: If `filename` already exists.
            ValueError: If both `stream` and `filename` are given.
        """

        if stream and filename:
            raise ValueError('only one of stream or filename can be given.')

        from collections import OrderedDict
        s = dict()
        s['scope'] = dict()
        s['scope']['name'] = self.name
        s['scope']['desc'] = self.desc
        s['inputs'] = {}
        s['outputs'] = {}

        const_keys = ['ptype','desc','dtype','default']
        parameter_keys = OrderedDict([
            # ('shortname', lambda x: x or None), # processed separately
            ('ptype', lambda x: x),
            ('desc', lambda x: x),
            ('dtype', lambda x: x),
            ('default', lambda x: x),
            ('min', lambda x: x),
            ('max', lambda x: x),
            # ('dist', lambda x: _name_or_dict(x) or None), # processed separately
            ('corr', lambda x: x or None),
            ('values', lambda x: x or None),
            ('abbrev', lambda x: x or None),
        ])
        measure_keys = {
            # 'shortname': lambda x: x or None,  # processed separately
            'kind':  lambda x: {-1:'minimize', 0:'info', 1:'maximize'}.get(x,x),
            'transform': lambda x: x,
            'metamodeltype': lambda x: 'linear' if x is None else x,
        }
        if strip_measure_transforms:
            measure_keys.pop('transform', None)

        for i in self._c_list:
            s['inputs'][i.name] = {}
            for k in const_keys:
                if hasattr(i, k):
                    v = getattr(i,k)
                    if v is not None:
                        s['inputs'][i.name][k] = getattr(i,k)

        for i in self._x_list + self._l_list:
            s['inputs'][i.name] = {}
            if i.shortname_if_any:
                s['inputs'][i.name]['shortname'] = i.shortname_if_any
            for k in parameter_keys:
                if hasattr(i, k):
                    v = parameter_keys[k](getattr(i,k))
                    if v is not None:
                        s['inputs'][i.name][k] = v
            v = i.distdef
            if v is not None:
                s['inputs'][i.name]['dist'] = _name_or_dict(v)

        for i in self._m_list:
            if include_measures is not None and i.name not in include_measures:
                continue
            if exclude_measures is not None and i.name in exclude_measures:
                continue
            s['outputs'][i.name] = {}
            if i.shortname_if_any:
                s['outputs'][i.name]['shortname'] = i.shortname_if_any
            for k in measure_keys:
                if hasattr(i, k):
                    s['outputs'][i.name][k] = measure_keys[k](getattr(i,k))

        import yaml.representer
        yaml.add_representer(dict,
                             lambda self, data: yaml.representer.SafeRepresenter.represent_dict(self, data.items()))

        if filename is not None:
            import os
            if os.path.exists(filename):
                raise FileExistsError(filename)
            with open(filename, 'w') as stream:
                yaml.dump(s, stream=stream, default_flow_style=default_flow_style, **kwargs)
        else:
            return yaml.dump(s, stream=stream, default_flow_style=default_flow_style, **kwargs)

    def info(self, return_string=False):
        """Print a summary of this Scope.

        Args:
            return_string (bool): Defaults False (print to stdout) but if given as True
                then this function returns the string instead of printing it.
        """

        if return_string:
            import io
            f = io.StringIO
        else:
            f = None

        print(f'name: {self.name}', file=f)
        print(f'desc: {self.desc}', file=f)
        if self._c_list:
            print('constants:', file=f)
            for i in self._c_list:
                print(f'  {i.name} = {i.default}', file=f)
        if self._x_list:
            print('uncertainties:', file=f)
            for i in self._x_list:
                if i.dtype in ('int','real'):
                    print(f'  {i.name} = {i.min} to {i.max}', file=f)
                elif i.dtype in ('bool',):
                    print(f'  {i.name} = boolean', file=f)
                elif i.dtype in ('cat',):
                    print(f'  {i.name} = categorical', file=f)
        if self._l_list:
            print('levers:', file=f)
            for i in self._l_list:
                if i.dtype in ('int','real'):
                    print(f'  {i.name} = {i.min} to {i.max}', file=f)
                elif i.dtype in ('bool',):
                    print(f'  {i.name} = boolean', file=f)
                elif i.dtype in ('cat',):
                    print(f'  {i.name} = categorical', file=f)
        if self._m_list:
            print('measures:', file=f)
            for i in self._m_list:
                print(f'  {i.name}', file=f)

        if return_string:
            return f.getvalue()

    def get_uncertainty_names(self):
        """Get a list of exogenous uncertainty names."""
        return [i.name for i in self._x_list]

    def _get_uncertainty_and_constant_names(self):
        """Get a list of exogenous uncertainty and constant names."""
        return self.get_uncertainty_names() + self.get_constant_names()

    def get_lever_names(self):
        """Get a list of policy lever names."""
        return [i.name for i in self._l_list]

    def get_constant_names(self):
        """Get a list of model constant names."""
        return [i.name for i in self._c_list]

    def get_parameter_names(self):
        """Get a list of model parameter (uncertainty+lever+constant) names."""
        return self.get_constant_names()+self.get_uncertainty_names()+self.get_lever_names()

    def get_all_names(self):
        """Get a list of all (uncertainty+lever+constant+measure) model names."""
        return self.get_parameter_names()+self.get_measure_names()

    def get_measure_names(self):
        """Get a list of performance measure names."""
        return [i.name for i in self._m_list]

    def get_uncertainties(self):
        """Get a list of exogenous uncertainties."""
        return [i for i in self._x_list]

    def get_levers(self):
        """Get a list of policy levers."""
        return [i for i in self._l_list]

    def get_constants(self):
        """Get a list of model constants."""
        return [i for i in self._c_list]

    def get_parameters(self):
        """Get a list of model parameters (uncertainties+levers+constants)."""
        return self.get_constants()+self.get_uncertainties()+self.get_levers()

    def get_measures(self):
        """Get a list of performance measures."""
        return [i for i in self._m_list]

    def __getitem__(self, item):
        """Get a parameter or measure by name."""
        for i in itertools.chain(self._x_list, self._l_list, self._c_list, self._m_list):
            if i.name == item:
                return i
        raise KeyError(item)

    def __contains__(self, item):
        for i in itertools.chain(self._x_list, self._l_list, self._c_list, self._m_list):
            if i.name == item:
                return True
        return False

    def ensure_dtypes(self, df):
        """
        Convert columns of dataframe to correct dtype as needed.

        Args:
            df (pandas.DataFrame): A dataframe with column names
                that are uncertainties, levers, or measures.

        Returns:
            pandas.DataFrame:
                The same data as input, but with dtypes as appropriate.
        """
        correct_dtypes = { }

        correct_dtypes.update({i.name: (i.dtype, getattr(i,'values',None)) for i in self.get_parameters()})
        correct_dtypes.update({i.name: (i.dtype, getattr(i,'values',None)) for i in self.get_measures()})

        for col in df.columns:
            if col in correct_dtypes:
                correct_dtype, cat_values = correct_dtypes[col]
                if correct_dtype == 'real':
                    df[col] = df[col].astype(float)
                elif correct_dtype == 'int':
                    df[col] = df[col].astype(int)
                elif correct_dtype == 'bool':
                    t = df[col].apply(lambda z: z.value if isinstance(z,Category) else z)
                    df[col] = t.astype(bool)
                elif correct_dtype == 'cat':
                    t = df[col].apply(lambda z: z.value if isinstance(z,Category) else z)
                    df[col] = pandas.Categorical(t, categories=cat_values, ordered=True)
                elif correct_dtype is None and df[col].dtype is numpy.dtype('O'):
                    df[col] = df[col].astype(float)

        return df

    def get_dtype(self, name):
        """
        Get the dtype for a parameter or measure.

        Args:
            name (str):
                The name of the parameter or measure

        Returns:
            str:
                {'real', 'int', 'bool', 'cat'}
        """
        correct_dtypes = { }

        correct_dtypes.update({i.name: i.dtype for i in self.get_parameters()})
        correct_dtypes.update({i.name: i.dtype for i in self.get_measures()})

        if name not in correct_dtypes:
            raise KeyError(name)
        return correct_dtypes[name]

    def get_cat_values(self, name):
        """
        Get the category values for a parameter or measure.

        Args:
            name (str):
                The name of the parameter or measure

        Returns:
            list or None
        """
        correct_dtypes = {}

        correct_dtypes.update({i.name: getattr(i,'values',None) for i in self.get_parameters()})
        correct_dtypes.update({i.name: getattr(i,'values',None) for i in self.get_measures()})

        if name not in correct_dtypes:
            raise KeyError(name)
        return correct_dtypes[name]

    def design_experiments(self, *args, **kwargs):
        """
        Create a design of experiments based on this Scope.

        Args:
            n_samples_per_factor (int, default 10): The number of samples in the
                design per random factor.
            n_samples (int or tuple, optional): The total number of samples in the
                design.  If `jointly` is False, this is the number of samples in each
                of the uncertainties and the levers, the total number of samples will
                be the square of this value.  Give a 2-tuple to set values for
                uncertainties and levers respectively, to set them independently.
                If this argument is given, it overrides `n_samples_per_factor`.
            random_seed (int or None, default 1234): A random seed for reproducibility.
            db (Database, optional): If provided, this design will be stored in the
                database indicated.
            design_name (str, optional): A name for this design, to identify it in the
                database. If not given, a unique name will be generated based on the
                selected sampler.  Has no effect if no `db` is given.
            sampler (str or AbstractSampler, default 'lhs'): The sampler to use for this
                design.  Available pre-defined samplers include:
                    - 'lhs': Latin Hypercube sampling
                    - 'ulhs': Uniform Latin Hypercube sampling, which ignores defined
                        distribution shapes from the scope and samples everything
                        as if it was from a uniform distribution
                    - 'mc': Monte carlo sampling
                    - 'uni': Univariate sensitivity testing, whereby experiments are
                        generated setting each parameter individually to minimum and
                        maximum values (for numeric dtypes) or all possible values
                        (for boolean and categorical dtypes).  Note that designs for
                        univariate sensitivity testing are deterministic and the number
                        of samples given is ignored.
            sample_from ('all', 'uncertainties', or 'levers'): Which scope components
                from which to sample.  Components not sampled are set at their default
                values in the design.
            jointly (bool, default True): Whether to sample jointly all uncertainties
                and levers in a single design, or, if False, to generate separate samples
                for levers and uncertainties, and then combine the two in a full-factorial
                manner.  This argument has no effect unless `sample_from` is 'all'.
                Note that jointly may produce a very large design;

        Returns:
            pandas.DataFrame: The resulting design.
        """
        if 'scope' in kwargs:
            kwargs.pop('scope')

        from ..experiment import experimental_design
        return experimental_design.design_experiments(self, *args, **kwargs)

    def _any_correlated_parameters(self):
        for p in self.get_parameters():
            if len(p.corr):
                return True
        return False

    def get_density(self, *args, **kwargs):
        """
        Compute the parametric density at any point.
        """
        if self._any_correlated_parameters():
            raise NotImplementedError("density with correlated parameters is coming soon")

        if args:
            for arg in args:
                kwargs.update(arg)

        density = 1.0
        for p in self.get_parameters():
            value = kwargs.get(p.name, p.default)
            density *= p.dist.pdf(value)
        return density

    def shortname(self, name):
        """
        Get a shortname, if available, for any named parameter or measure.

        Args:
            name: str
        Returns:
            str
        """
        try:
            x = self[name]
        except KeyError:
            return name
        else:
            try:
                return x.shortname
            except:
                return name

    def get_description(self, name):
        """
        Get a description, if available, for any named parameter or measure.

        Args:
            name: str
        Returns:
            str
        """
        try:
            x = self[name]
        except KeyError:
            return ''
        else:
            try:
                return x.desc
            except:
                return ''

    def default_policy(self, **kwargs):
        """
        The default settings for policy levers.

        Args:
            **kwargs:
                Override the defaults given in the scope
                with these values.

        Returns:
            ema_workbench.Policy
        """
        from ema_workbench import Policy
        values = {l.name: l.default for l in self.get_levers()}
        values.update(kwargs)
        return Policy('default', **values)

    def default_scenario(self, **kwargs):
        """
        The default settings for exogenous uncertainties.

        Args:
            **kwargs:
                Override the defaults given in the scope
                with these values.

        Returns:
            ema_workbench.Scenario
        """
        from ema_workbench import Scenario
        values = {u.name: u.default for u in self.get_uncertainties()}
        values.update(kwargs)
        return Scenario('default', **values)
