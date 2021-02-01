# -*- coding: utf-8 -*-

import numpy as np

class Dummy():

    
    def __init__(self):
        '''nothing to do here'''

    def calc_pm1(self, exp_vars):
        return self.__sum_exp_vars(exp_vars)

    def calc_pm2(self, exp_vars):
        return self.__sum_exp_vars(exp_vars) * 2

    def calc_pm10(self, exp_vars):
        return self.__sum_exp_vars(exp_vars) * 10
    
    def calc_pm100(self, exp_vars):
        return self.__sum_exp_vars(exp_vars) * 100    
    
    def __sum_exp_vars(self,ev):
        return ev['exp_var 1'] + ev['exp_var 2']
        

    def __call__(self, **kwargs):
        return dict(
            pm_1=self.calc_pm1(kwargs),
            pm_2=self.calc_pm2(kwargs),
            pm_10=self.calc_pm10(kwargs),
            pm_100=self.calc_pm100(kwargs),
        )



def NoisyDummy(**kwargs):
    lever1 = kwargs.get('lever1', 0)
    lever2 = kwargs.get('lever2', 0)
    uncertain1 = kwargs.get('uncertain1', 3)
    uncertain2 = np.exp(kwargs.get('uncertain2', -0.7))
    uncertain3 = np.exp(kwargs.get('uncertain3', 0.7))
    certain4 = kwargs.get('certain4', 3)

    noise_amplitude = kwargs.get('noise_amplitude', 2.0)
    noise_frequency = kwargs.get('noise_frequency', 5.0)

    pm_1 = (
        - uncertain2 * lever1 * lever1
        + (uncertain1 + certain4) * (lever1 + lever2)
        + noise_amplitude * np.sin(noise_frequency * lever1)
    )

    pm_2 = np.minimum(
        1.11e+111 * uncertain1,
        np.exp(
            uncertain3 * lever1 * (lever1 + lever2)
            + uncertain1 * lever1
            + noise_amplitude * np.cos(noise_frequency * lever2)
        )
    )

    pm_3 = (
        noise_amplitude * np.cos(noise_frequency * lever1)
        + noise_amplitude * np.sin(noise_frequency * lever2)
        + certain4
    )

    pm_4 = np.exp(
            uncertain1 + certain4
    )

    return {'pm_1':pm_1, 'pm_2': pm_2, 'pm_3': pm_3, 'pm_4':pm_4}


def pmt(rate, nper, pv, fv=0, when='end'):
    """
    Compute the payment against loan principal plus interest.

    Given:
     * a present value, `pv` (e.g., an amount borrowed)
     * a future value, `fv` (e.g., 0)
     * an interest `rate` compounded once per period, of which
       there are
     * `nper` total
     * and (optional) specification of whether payment is made
       at the beginning (`when` = {'begin', 1}) or the end
       (`when` = {'end', 0}) of each period

    Return:
       the (fixed) periodic payment.

    Parameters
    ----------
    rate : array_like
        Rate of interest (per period)
    nper : array_like
        Number of compounding periods
    pv : array_like
        Present value
    fv : array_like,  optional
        Future value (default = 0)
    when : {{'begin', 1}, {'end', 0}}, {string, int}
        When payments are due ('begin' (1) or 'end' (0))

    Returns
    -------
    out : ndarray
        Payment against loan plus interest.  If all input is scalar, returns a
        scalar float.  If any input is array_like, returns payment for each
        input element. If multiple inputs are array_like, they all must have
        the same shape.

    Notes
    -----
    This function is replicated from the numpy_financial package, under the same
    LICENSE as the TMIP-EMAT repository.
    Copyright (c) 2005-2019, NumPy Developers. All rights reserved.

    """

    _when_to_num = {'end': 0, 'begin': 1,
                    'e': 0, 'b': 1,
                    0: 0, 1: 1,
                    'beginning': 1,
                    'start': 1,
                    'finish': 0}

    def _convert_when(when):
        # Test to see if when has already been converted to ndarray
        # This will happen if one function calls another, for example ppmt
        if isinstance(when, np.ndarray):
            return when
        try:
            return _when_to_num[when]
        except (KeyError, TypeError):
            return [_when_to_num[x] for x in when]
    when = _convert_when(when)
    (rate, nper, pv, fv, when) = map(np.array, [rate, nper, pv, fv, when])
    temp = (1 + rate)**nper
    mask = (rate == 0)
    masked_rate = np.where(mask, 1, rate)
    fact = np.where(mask != 0, nper,
                    (1 + masked_rate*when)*(temp - 1)/masked_rate)
    return -(fv + pv*temp) / fact

def Road_Capacity_Investment(
        # constant
        free_flow_time=60,
        initial_capacity=100,

        # uncertainty
        alpha=0.15,
        beta=4.0,
        input_flow=100,
        value_of_time=0.01,
        unit_cost_expansion=1,
        interest_rate=0.03,
        yield_curve=0.01,

        # policy
        expand_capacity=10,
        amortization_period=30,
        interest_rate_lock=False,
        debt_type='GO Bond',
        lane_width=10,

        **kwargs,
):
    """
    A fictitious example model for road capacity investment.

    This model simulates a capacity expansion investment on a single
    network link.  The link volume-delay function is governed by the
    `BPR function <https://en.wikipedia.org/wiki/Route_assignment#Frank-Wolfe_algorithm>`_.

    This model is a bit contrived, because it is designed to explicitly demonstrate
    a wide variety of EMAT features in a transportation planning model that is as simple
    as possible.  For example, the policy levers are structured so that there is one
    of each dtype (float, int, bool, and categorical).

    Args:
        free_flow_time (float, default 60): The free flow travel time on the link.
        initial_capacity (float, default 100): The pre-expansion capacity on the link.
        alpha (float, default 0.15): Alpha parameter to the BPR volume-delay function.
        beta (float, default 4.0): Beta parameter to the BPR volume-delay function.
        input_flow (float, default 100): The future input flow on the link.
        value_of_time (float, default 0.01): The value of a unit of travel time savings
            per unit of flow on the link.
        unit_cost_expansion (float, default 1): The present marginal cost of adding one
            unit of capacity to the link (assumes no economies of scale on expansion cost)
        interest_rate (float, default 0.03): The interest rate actually incurred for
            revenue bonds amortized over 15 years.  The interest rate for general obligation
            bonds is assumed to be 0.0025 less than this value.
        yield_curve (float, default 0.01): The marginal increase in the interest_rate if
            the amortization period is 50 years instead of 15.  The yield curve is assumed
            to be linearly projected to all other possible amortization periods
        expand_capacity (float, default 10): The amount of capacity expansion actually
            constructed.
        amortization_period (int, default 30): The time period over which the construction
            costs are amortized.
        interest_rate_lock (bool, default False): Whether interest rates are locked at
            the assumed current rate of 0.03 / 0.01 or allowed to float.
        debt_type ('GO Bond', 'Rev Bond', 'Paygo'): Type of financing.  General obligation
            bonds are assumed to have a lower interest rate than revenue bonds, but
            may be politically less desirable.  Pay-as-you-go financing incurs no actual
            interest costs, but requires actually having the funds available.
        lane_width (float, default 10): The width of lanes on the roadway.  This parameter
            is intentionally wacky, causing massive congestion for any value other than 10,
            to demonstrate what might happen with broken model inputs.

    Returns:
        dict:
            no_build_travel_time
                The average travel time on the link if no
                capacity expansion was constructed.
            build_travel_time
                The average travel time on the link after expansion.
            time_savings
                The average travel time savings as a result of the
                expansion.
            value_of_time_savings
                The total value of the travel time savings,
                accounting for the time savings per traveler, the total flow, and
                the value of time.
            present_cost_expansion
                The present cost of building the expansion
            cost_of_capacity_expansion
                The annual payment to finance the expansion,
                when amortized.
            net_benefits
                The value of the time savings minus the annual payment.



    """

    debt_type = debt_type.lower()
    assert debt_type in ('go bond', 'paygo', 'rev bond')

    average_travel_time0 = free_flow_time * (1 + alpha*(input_flow/initial_capacity)**beta)
    capacity = initial_capacity + expand_capacity
    average_travel_time1 = free_flow_time * (1 + alpha*(input_flow/capacity)**beta)
    oops = np.absolute(lane_width-10)
    average_travel_time1 += (oops*1000)**0.5 + np.sin(input_flow)*oops*2
    travel_time_savings = average_travel_time0 - average_travel_time1
    value_of_time_savings = value_of_time * travel_time_savings * input_flow
    present_cost_of_capacity_expansion = float(unit_cost_expansion * expand_capacity)

    if interest_rate_lock:
        interest_rate = 0.03
        yield_curve = 0.01

    if (debt_type == 'go bond'):
        interest_rate -= 0.0025
    elif (debt_type == 'paygo'):
        interest_rate = 0

    effective_interest_rate = interest_rate + yield_curve * (amortization_period-15) / 35

    cost_of_capacity_expansion = pmt(
        effective_interest_rate,
        amortization_period,
        present_cost_of_capacity_expansion,
    )

    return dict(
        no_build_travel_time=average_travel_time0,
        build_travel_time=average_travel_time1,
        time_savings=travel_time_savings,
        value_of_time_savings=value_of_time_savings,
        present_cost_expansion=present_cost_of_capacity_expansion,
        cost_of_capacity_expansion=-cost_of_capacity_expansion,
        net_benefits = value_of_time_savings + cost_of_capacity_expansion,
    )



def _Road_Capacity_Investment_CmdLine():
    """
    This is a demo for calling a core model function on the command line.
    """
    import argparse, pandas, os, sys, warnings
    parser = argparse.ArgumentParser()
    parser.add_argument('--levers', type=str, default='levers.yml', help='Levers Yaml File')
    parser.add_argument('--uncs', type=str, default="uncs.yml", help='Uncertainties Yaml File')
    parser.add_argument('--no-random-crashes', action='store_true', help='disable random crashes')

    args = parser.parse_args()
    import logging
    logger = logging.getLogger('emat.RoadTest')
    file_handler = logging.FileHandler("emat-road-test.log")
    file_handler.setLevel(10)
    LOG_FORMAT = '[%(asctime)s] %(name)s.%(levelname)s: %(message)s'
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(20)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(10)

    logger.info("running emat-road-test-demo")
    logger.debug(str(args))
    logger.debug(str(os.getcwd()))

    import yaml
    if os.path.exists(args.levers):
        with open(args.levers, 'rt') as f:
            levers = yaml.safe_load(f)
    else:
        levers = {'mandatory_unused_lever':42}
    if os.path.exists(args.uncs):
        with open(args.uncs, 'rt') as f:
            uncs = yaml.safe_load(f)
    else:
        uncs = {}

    if 'mandatory_unused_lever' not in levers:
        raise ValueError("missing 'mandatory_unused_lever'")
    if levers['mandatory_unused_lever'] != 42:
        raise ValueError("incorrect value for 'mandatory_unused_lever', must be 42")

    if 'unit_cost_expansion' in uncs:
        raise ValueError("cannot give 'unit_cost_expansion', use 'labor_unit_cost_expansion' and 'materials_unit_cost_expansion'")
    if uncs.get('labor_unit_cost_expansion', 0) <= uncs.get('materials_unit_cost_expansion', 0):
        raise ValueError("'labor_unit_cost_expansion' cannot be less than or equal 'materials_unit_cost_expansion'")
    if uncs.get('labor_unit_cost_expansion', 0) > uncs.get('materials_unit_cost_expansion', 0)*2:
        raise ValueError("'labor_unit_cost_expansion' cannot be more than double 'materials_unit_cost_expansion'")
    unit_cost_expansion = uncs.pop('labor_unit_cost_expansion', 0) + uncs.pop('materials_unit_cost_expansion', 0)
    uncs['unit_cost_expansion'] = unit_cost_expansion

    # (pseudo)random crash
    if not args.no_random_crashes:
        if 'expand_capacity' in levers and levers['expand_capacity'] > 90 and not os.path.exists('prevent_random_crash.txt'):
            with open('prevent_random_crash.txt', 'wt') as f:
                f.write("this file will prevent random crashes in `emat-road-test-demo`")
            logger.error("Random crash, ha ha!")
            sys.exit(-9)

    try:
        for k,v in levers.items():
            logger.debug(f"lever: {k} = {v}")
        for k,v in uncs.items():
            logger.debug(f"uncertainty: {k} = {v}")

        result = Road_Capacity_Investment(**levers, **uncs)
        for k,v in result.items():
            logger.debug(f"result: {k} = {v}")

        result1 = {str(k):float(result[k]) for k in ['no_build_travel_time','build_travel_time','time_savings']}
        result2 = pandas.DataFrame({
            'value_of_time_savings': [np.exp(result['value_of_time_savings']/1000), np.nan],
            'present_cost_expansion': [np.nan, result['present_cost_expansion']],
            'cost_of_capacity_expansion': [np.exp(result['cost_of_capacity_expansion']/1000), np.nan],
            'net_benefits': [np.nan,result['net_benefits']],
        }, index=['exp','plain'])

        with open('output.yaml', 'wt') as f:
            yaml.safe_dump(result1, f)
        result2.to_csv('output.csv.gz')

        logger.info("emat-road-test-demo completed without errors")
    except:
        logger.exception("unintentional crash")
        sys.exit(-8)
