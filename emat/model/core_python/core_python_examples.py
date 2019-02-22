# -*- coding: utf-8 -*-

import numpy

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
    uncertain2 = numpy.exp(kwargs.get('uncertain2', -0.7))
    uncertain3 = numpy.exp(kwargs.get('uncertain3', 0.7))
    certain4 = kwargs.get('certain4', 3)

    noise_amplitude = kwargs.get('noise_amplitude', 2.0)
    noise_frequency = kwargs.get('noise_frequency', 5.0)

    pm_1 = (
        - uncertain2 * lever1 * lever1
        + (uncertain1 + certain4) * (lever1 + lever2)
        + noise_amplitude * numpy.sin(noise_frequency * lever1)
    )

    pm_2 = numpy.minimum(
        1.11e+111 * uncertain1,
        numpy.exp(
            uncertain3 * lever1 * (lever1 + lever2)
            + uncertain1 * lever1
            + noise_amplitude * numpy.cos(noise_frequency * lever2)
        )
    )

    pm_3 = (
        noise_amplitude * numpy.cos(noise_frequency * lever1)
        + noise_amplitude * numpy.sin(noise_frequency * lever2)
        + certain4
    )

    pm_4 = numpy.exp(
            uncertain1 + certain4
    )

    return {'pm_1':pm_1, 'pm_2': pm_2, 'pm_3': pm_3, 'pm_4':pm_4}



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
    travel_time_savings = average_travel_time0 - average_travel_time1
    value_of_time_savings = value_of_time * travel_time_savings * input_flow
    present_cost_of_capacity_expansion = unit_cost_expansion * expand_capacity

    if interest_rate_lock:
        interest_rate = 0.03
        yield_curve = 0.01

    if (debt_type == 'go bond'):
        interest_rate -= 0.0025
    elif (debt_type == 'paygo'):
        interest_rate = 0

    effective_interest_rate = interest_rate + yield_curve * (amortization_period-15) / 35

    cost_of_capacity_expansion = numpy.pmt(effective_interest_rate,
                                           amortization_period,
                                           present_cost_of_capacity_expansion, )

    return dict(
        no_build_travel_time=average_travel_time0,
        build_travel_time=average_travel_time1,
        time_savings=travel_time_savings,
        value_of_time_savings=value_of_time_savings,
        present_cost_expansion=present_cost_of_capacity_expansion,
        cost_of_capacity_expansion=-cost_of_capacity_expansion,
        net_benefits = value_of_time_savings + cost_of_capacity_expansion,
    )

