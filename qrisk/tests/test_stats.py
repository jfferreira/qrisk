from __future__ import division

from unittest import TestCase
from nose_parameterized import parameterized
from numpy.testing import assert_almost_equal
import random

import numpy as np
import pandas as pd

import qrisk

DECIMAL_PLACES = 8


class TestStats(TestCase):

    # Simple benchmark, no drawdown
    simple_benchmark = pd.Series(
        np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]) / 100,
        index=pd.date_range('2000-1-30', periods=9, freq='D'))

    # All positive returns, small variance
    positive_returns = pd.Series(
        np.array([1., 2., 1., 1., 1., 1., 1., 1., 1.]) / 100,
        index=pd.date_range('2000-1-30', periods=9, freq='D'))

    # All negative returns
    negative_returns = pd.Series(
        np.array([0., -6., -7., -1., -9., -2., -6., -8., -5.]) / 100,
        index=pd.date_range('2000-1-30', periods=9, freq='D'))

    # Positive and negative returns with max drawdown
    mixed_returns = pd.Series(
        np.array([np.nan, 1., 10., -4., 2., 3., 2., 1., -10.]) / 100,
        index=pd.date_range('2000-1-30', periods=9, freq='D'))

    # Weekly returns
    weekly_returns = pd.Series(
        np.array([0., 1., 10., -4., 2., 3., 2., 1., -10.])/100,
        index=pd.date_range('2000-1-30', periods=9, freq='W'))

    # Monthly returns
    monthly_returns = pd.Series(
        np.array([0., 1., 10., -4., 2., 3., 2., 1., -10.])/100,
        index=pd.date_range('2000-1-30', periods=9, freq='M'))

    # Series of length 1
    one_return = pd.Series(
        np.array([1.])/100,
        index=pd.date_range('2000-1-30', periods=1, freq='D'))

    # Empty series
    empty_returns = pd.Series(
        np.array([])/100,
        index=pd.date_range('2000-1-30', periods=0, freq='D'))

    # Random noise
    noise = pd.Series(
        [random.gauss(0, 0.001) for i in range(1000)],
        index=pd.date_range('2000-1-30', periods=1000, freq='D')
    )
    noise_uniform = pd.Series(
        [random.uniform(-0.01, 0.01) for i in range(1000)],
        index=pd.date_range('2000-1-30', periods=1000, freq='D')
    )

    # Random noise inv
    inv_noise = noise.multiply(-1)

    # Flat line
    flat_line_0 = pd.Series(
        np.linspace(0, 0, num=1000),
        index=pd.date_range('2000-1-30', periods=1000, freq='D')
        )
    # Flat line
    flat_line_1 = pd.Series(
        np.linspace(0.01, 0.01, num=1000),
        index=pd.date_range('2000-1-30', periods=1000, freq='D')
        )

    # Positive line
    pos_line = pd.Series(
        np.linspace(0, 1, num=1000),
        index=pd.date_range('2000-1-30', periods=1000, freq='D')
    )

    # Negative line
    neg_line = pd.Series(
        np.linspace(0, -1, num=1000),
        index=pd.date_range('2000-1-30', periods=1000, freq='D')
    )

    one = [-0.00171614, 0.01322056, 0.03063862, -0.01422057, -0.00489779,
           0.01268925, -0.03357711, 0.01797036]
    two = [0.01846232, 0.00793951, -0.01448395, 0.00422537, -0.00339611,
           0.03756813, 0.0151531, 0.03549769]

    df_index_simple = pd.date_range('2000-1-30', periods=8, freq='D')
    df_index_week = pd.date_range('2000-1-30', periods=8, freq='W')
    df_index_month = pd.date_range('2000-1-30', periods=8, freq='M')

    df_simple = pd.DataFrame({
        'one': pd.Series(one, index=df_index_simple),
        'two': pd.Series(two, index=df_index_simple)})

    df_week = pd.DataFrame({
        'one': pd.Series(one, index=df_index_week),
        'two': pd.Series(two, index=df_index_week)})

    df_month = pd.DataFrame({
        'one': pd.Series(one, index=df_index_month),
        'two': pd.Series(two, index=df_index_month)})

    @parameterized.expand([
        (mixed_returns, 0, [0.0, 0.01, 0.111, 0.066559, 0.08789, 0.12052,
                            0.14293, 0.15436, 0.03893]),
        (mixed_returns, 100, [100.0, 101.0, 111.1, 106.65599, 108.78912,
                              112.05279, 114.29384, 115.43678, 103.89310]),
        (negative_returns, 0, [0.0, -0.06, -0.1258, -0.13454, -0.21243,
                               -0.22818, -0.27449, -0.33253, -0.36590])
    ])
    def test_cum_returns(self, returns, starting_value, expected):
        cum_returns = qrisk.cum_returns(returns, starting_value=starting_value)
        for i in range(returns.size):
            assert_almost_equal(
                cum_returns[i],
                expected[i],
                4)

    @parameterized.expand([
        (simple_benchmark, qrisk.WEEKLY, [0.010000000000000009,
                                          0.072135352107010053,
                                          0.010000000000000009]),
        (simple_benchmark, qrisk.MONTHLY, [0.020100000000000007,
                                           0.072135352107010053]),
        (simple_benchmark, qrisk.YEARLY, [0.093685272684361109]),
        (weekly_returns, qrisk.MONTHLY, [0.0, 0.087891200000000058,
                                         -0.04500459999999995]),
        (weekly_returns, qrisk.YEARLY, [0.038931091700480147]),
        (monthly_returns, qrisk.YEARLY, [0.038931091700480147])
    ])
    def test_aggregate_returns(self, returns, convert_to, expected):
        returns = qrisk.aggregate_returns(returns, convert_to).values.tolist()
        for i, v in enumerate(returns):
            assert_almost_equal(
                v,
                expected[i],
                DECIMAL_PLACES)

    @parameterized.expand([
        (empty_returns, np.nan),
        (one_return, 0.0),
        (simple_benchmark, 0.0),
        (mixed_returns, -0.1),
        (positive_returns, -0.0),
        (negative_returns, -0.36590730349873601),
        (pd.Series(
            np.array([10, -10, 10]) / 100,
            index=pd.date_range('2000-1-30', periods=3, freq='D')),
            -0.10)
    ])
    def test_max_drawdown(self, returns, expected):
        assert_almost_equal(
            qrisk.max_drawdown(
                returns
            ),
            expected,
            DECIMAL_PLACES)

    # Multiplying returns by a positive constant larger than 1 will increase
    # the maximum drawdown by a factor greater than or equal to the constant.
    # Similarly, a positive constant smaller than 1 will decrease maximum
    # drawdown by at least the constant.
    @parameterized.expand([
        (noise_uniform, 1.1),
        (noise, 2),
        (noise_uniform, 10),
        (noise_uniform, 0.99),
        (noise, 0.5)
    ])
    def test_max_drawdown_transformation(self, returns, constant):
        max_dd = qrisk.max_drawdown(returns)
        transformed_dd = qrisk.max_drawdown(constant*returns)
        if constant >= 1:
            assert constant*max_dd <= transformed_dd
        else:
            assert constant*max_dd >= transformed_dd

    # Translating returns by a positive constant should increase the maximum
    # drawdown to a maximum of zero. Translating by a negative constant
    # decreases the maximum drawdown.
    @parameterized.expand([
        (noise, .0001),
        (noise, .001),
        (noise_uniform, .01),
        (noise_uniform, .1),
    ])
    def test_max_drawdown_translation(self, returns, constant):
        depressed_returns = returns-constant
        raised_returns = returns+constant
        max_dd = qrisk.max_drawdown(returns)
        depressed_dd = qrisk.max_drawdown(depressed_returns)
        raised_dd = qrisk.max_drawdown(raised_returns)
        assert max_dd <= raised_dd
        assert depressed_dd <= max_dd

    @parameterized.expand([
        (mixed_returns, qrisk.DAILY, 1.9135925373194231),
        (weekly_returns, qrisk.WEEKLY, 0.24690830513998208),
        (monthly_returns, qrisk.MONTHLY, 0.052242061386048144)
    ])
    def test_annual_ret(self, returns, period, expected):
        assert_almost_equal(
            qrisk.annual_return(
                returns,
                period=period
            ),
            expected,
            DECIMAL_PLACES)

    @parameterized.expand([
        (simple_benchmark, qrisk.DAILY, 0.0),
        (mixed_returns, qrisk.DAILY, 0.8552777326693359),
        (weekly_returns, qrisk.WEEKLY, 0.38851569394870583),
        (monthly_returns, qrisk.MONTHLY, 0.18663690238892558)
    ])
    def test_annual_volatility(self, returns, period, expected):
        assert_almost_equal(
            qrisk.annual_volatility(
                returns,
                period=period
            ),
            expected,
            DECIMAL_PLACES
        )

    @parameterized.expand([
        (empty_returns, qrisk.DAILY, np.nan),
        (one_return, qrisk.DAILY, np.nan),
        (mixed_returns, qrisk.DAILY, 19.135925373194233),
        (weekly_returns, qrisk.WEEKLY, 2.4690830513998208),
        (monthly_returns, qrisk.MONTHLY, 0.52242061386048144)
    ])
    def test_calmar(self, returns, period, expected):
        assert_almost_equal(
            qrisk.calmar_ratio(
                returns,
                period=period
            ),
            expected,
            DECIMAL_PLACES)

    # Regression tests for omega ratio
    @parameterized.expand([
        (empty_returns, 0.0, 0.0, np.nan),
        (one_return, 0.0, 0.0, np.nan),
        (mixed_returns, 0.0, 10.0, 0.78629772289706013),
        (mixed_returns, 0.0, -10.0, np.nan),
        (mixed_returns, simple_benchmark, 0.0, 0.76470588235294112),
        (positive_returns, 0.01, 0.0, np.nan),
        (positive_returns, 0.011, 0.0, 1.125),
        (positive_returns, 0.02, 0.0, 0.0),
        (negative_returns, 0.01, 0.0, 0.0)
    ])
    def test_omega(self, returns, risk_free, required_return,
                   expected):
        assert_almost_equal(
            qrisk.omega_ratio(
                returns,
                risk_free=risk_free,
                required_return=required_return),
            expected,
            DECIMAL_PLACES)

    # As the required return increases (but is still less than the maximum
    # return), omega decreases
    @parameterized.expand([
        (noise_uniform, 0.0, 0.001),
        (noise, .001, .002),
    ])
    def test_omega_returns(self, returns, required_return_less,
                           required_return_more):
        assert qrisk.omega_ratio(returns, required_return_less) > \
            qrisk.omega_ratio(returns, required_return_more)

    # Regressive sharpe ratio tests
    @parameterized.expand([
        (empty_returns, 0.0, np.nan),
        (one_return, 0.0, np.nan),
        (mixed_returns, mixed_returns, np.nan),
        (mixed_returns, 0.0, 1.6368951821422701),
        (mixed_returns, simple_benchmark, -1.3095161457138154),
        (positive_returns, 0.0, 52.915026221291804),
        (negative_returns, 0.0, -24.406808633910085)
    ])
    def test_sharpe_ratio(self, returns, risk_free, expected):
        assert_almost_equal(
            qrisk.sharpe_ratio(
                np.asarray(returns),
                risk_free=risk_free),
            expected,
            DECIMAL_PLACES)

    # Translating the returns and required returns by the same amount
    # should not change the sharpe ratio.
    @parameterized.expand([
        (noise_uniform, 0, .005),
        (noise_uniform, 0.005, .005)
    ])
    def test_sharpe_translation(self, returns, required_return, translation):
        sr = qrisk.sharpe_ratio(returns, required_return)
        sr_depressed = qrisk.sharpe_ratio(
            returns-translation,
            required_return-translation)
        sr_raised = qrisk.sharpe_ratio(
            returns+translation,
            required_return+translation)
        assert_almost_equal(
            sr,
            sr_depressed,
            DECIMAL_PLACES)
        assert_almost_equal(
            sr,
            sr_raised,
            DECIMAL_PLACES)

    # Translating the required return inversely affects the sharpe ratio.
    @parameterized.expand([
        (noise_uniform, 0, .005),
        (noise, 0, .005)
    ])
    def test_sharpe_translation_1(self, returns, required_return, translation):
        sr = qrisk.sharpe_ratio(returns, required_return)
        sr_depressed = qrisk.sharpe_ratio(
            returns,
            required_return-translation)
        sr_raised = qrisk.sharpe_ratio(
            returns,
            required_return+translation)
        assert sr_depressed > sr
        assert sr > sr_raised

    # Returns of a wider range or larger standard deviation decreases the
    # sharpe ratio
    @parameterized.expand([
        (.001, .002),
        (.01, .02)
    ])
    def test_sharpe_noise(self, small, large):
        index = pd.date_range('2000-1-30', periods=1000, freq='D')
        smaller_normal = pd.Series(
            [random.gauss(.01, small) for i in range(1000)],
            index=index
        )
        larger_normal = pd.Series(
            [random.gauss(.01, large) for i in range(1000)],
            index=index
        )
        assert qrisk.sharpe_ratio(smaller_normal, 0.001) > \
            qrisk.sharpe_ratio(larger_normal, 0.001)

    # Regressive downside risk tests
    @parameterized.expand([
        (empty_returns, 0.0, qrisk.DAILY, np.nan),
        (one_return, 0.0, qrisk.DAILY, 0.0),
        (mixed_returns, mixed_returns, qrisk.DAILY, 0.0),
        (mixed_returns, 0.0, qrisk.DAILY, 0.5699122739510003),
        (mixed_returns, 0.1, qrisk.DAILY, 1.7023513150933332),
        (weekly_returns, 0.0, qrisk.WEEKLY, 0.25888650451930134),
        (weekly_returns, 0.1, qrisk.WEEKLY, 0.7733045971672482),
        (monthly_returns, 0.0, qrisk.MONTHLY, 0.1243650540411842),
        (monthly_returns, 0.1, qrisk.MONTHLY, 0.37148351242013422),
        (df_simple, 0.0, qrisk.DAILY,
         pd.Series([0.20671788246185202, 0.083495680595704475],
                   index=['one', 'two'])),
        (df_week, 0.0, qrisk.WEEKLY,
         pd.Series([0.093902996054410062, 0.037928477556776516],
                   index=['one', 'two'])),
        (df_month, 0.0, qrisk.MONTHLY,
         pd.Series([0.045109540184877193, 0.018220251263412916],
                   index=['one', 'two']))
    ])
    def test_downside_risk(self, returns, required_return, period, expected):
        downside_risk = qrisk.downside_risk(
                        returns,
                        required_return=required_return,
                        period=period)
        if isinstance(downside_risk, float):
            assert_almost_equal(
                downside_risk,
                expected,
                DECIMAL_PLACES)
        else:
            for i in range(downside_risk.size):
                assert_almost_equal(
                    downside_risk[i],
                    expected[i],
                    DECIMAL_PLACES)

    # As a higher percentage of returns are below the required return,
    # downside risk increases.
    @parameterized.expand([
        (noise, flat_line_0),
        (noise_uniform, flat_line_0)
    ])
    def test_downside_risk_noisy(self, noise, flat_line):
        noisy_returns_1 = noise[0:250].add(flat_line[250:], fill_value=0)
        noisy_returns_2 = noise[0:500].add(flat_line[500:], fill_value=0)
        noisy_returns_3 = noise[0:750].add(flat_line[750:], fill_value=0)
        dr_1 = qrisk.downside_risk(noisy_returns_1, flat_line)
        dr_2 = qrisk.downside_risk(noisy_returns_2, flat_line)
        dr_3 = qrisk.downside_risk(noisy_returns_3, flat_line)
        assert dr_1 <= dr_2
        assert dr_2 <= dr_3

    # Downside risk increases as the required_return increases
    @parameterized.expand([
        (noise, .005),
        (noise_uniform, .005)
    ])
    def test_downside_risk_trans(self, returns, required_return):
        dr_0 = qrisk.downside_risk(returns, -required_return)
        dr_1 = qrisk.downside_risk(returns, 0)
        dr_2 = qrisk.downside_risk(returns, required_return)
        assert dr_0 <= dr_1
        assert dr_1 <= dr_2

    # Downside risk for a random series with a required return of 0 is higher
    # for datasets with larger standard deviation
    @parameterized.expand([
        (.001, .002),
        (.001, .01),
        (0, .001)
    ])
    def test_downside_risk_std(self, smaller_std, larger_std):
        less_noise = pd.Series(
            [random.gauss(0, smaller_std) for i in range(1000)],
            index=pd.date_range('2000-1-30', periods=1000, freq='D')
        )
        more_noise = pd.Series(
            [random.gauss(0, larger_std) for i in range(1000)],
            index=pd.date_range('2000-1-30', periods=1000, freq='D')
        )
        assert qrisk.downside_risk(less_noise) < \
            qrisk.downside_risk(more_noise)

    # Regressive sortino ratio tests
    @parameterized.expand([
        (empty_returns, 0.0, qrisk.DAILY, np.nan),
        (one_return, 0.0, qrisk.DAILY, np.nan),
        (mixed_returns, mixed_returns, qrisk.DAILY, np.nan),
        (mixed_returns, 0.0, qrisk.DAILY, 2.456518422202588),
        (mixed_returns, simple_benchmark, qrisk.DAILY, -1.7457431218879385),
        (positive_returns, 0.0, qrisk.DAILY, np.inf),
        (negative_returns, 0.0, qrisk.DAILY, -13.532743075043401),
        (simple_benchmark, 0.0, qrisk.DAILY, np.inf),
        (weekly_returns, 0.0, qrisk.WEEKLY, 0.50690062680370862),
        (monthly_returns, 0.0, qrisk.MONTHLY, 0.11697706772393276),
        (df_simple, 0.0, qrisk.DAILY,
         pd.Series([3.0639640966566306, 38.090963117002495],
                   index=['one', 'two'])),
        (df_week, 0.0, qrisk.WEEKLY,
         pd.Series([0.63224655962755871, 7.8600400082703556],
                   index=['one', 'two'])),
        (df_month, 0.0, qrisk.MONTHLY,
         pd.Series([0.14590305222174432, 1.8138553865239282],
                   index=['one', 'two']))
    ])
    def test_sortino(self, returns, required_return, period, expected):
        sortino_ratio = qrisk.sortino_ratio(
                        returns,
                        required_return=required_return,
                        period=period)
        if isinstance(sortino_ratio, float):
            assert_almost_equal(
                sortino_ratio,
                expected,
                DECIMAL_PLACES)
        else:
            for i in range(sortino_ratio.size):
                assert_almost_equal(
                    sortino_ratio[i],
                    expected[i],
                    DECIMAL_PLACES)

    # A large Sortino ratio indicates there is a low probability of a large
    # loss, therefore randomly changing values larger than required return to a
    # loss of 25 percent decreases the ratio.
    @parameterized.expand([
        (noise_uniform, 0),
        (noise, 0),
    ])
    def test_sortino_add_noise(self, returns, required_return):
        sr_1 = qrisk.sortino_ratio(returns, required_return)
        upside_values = returns[returns > required_return].index.tolist()
        # Add large losses at random upside locations
        loss_loc = random.sample(upside_values, 2)
        returns[loss_loc[0]] = -0.01
        sr_2 = qrisk.sortino_ratio(returns, required_return)
        returns[loss_loc[1]] = -0.01
        sr_3 = qrisk.sortino_ratio(returns, required_return)
        assert sr_1 > sr_2
        assert sr_2 > sr_3

    # Similarly, randomly increasing some values below the required return to
    # the required return increases the ratio.
    @parameterized.expand([
        (noise_uniform, 0),
        (noise, 0)
    ])
    def test_sortino_sub_noise(self, returns, required_return):
        sr_1 = qrisk.sortino_ratio(returns, required_return)
        downside_values = returns[returns < required_return].index.tolist()
        # Replace some values below the required return to the required return
        loss_loc = random.sample(downside_values, 2)
        returns[loss_loc[0]] = required_return
        sr_2 = qrisk.sortino_ratio(returns, required_return)
        returns[loss_loc[1]] = required_return
        sr_3 = qrisk.sortino_ratio(returns, required_return)
        assert sr_1 <= sr_2
        assert sr_2 <= sr_3

    # Translating the returns and required returns by the same amount
    # should not change the sortino ratio.
    @parameterized.expand([
        (noise_uniform, 0, .005),
        (noise_uniform, 0.005, .005)
    ])
    def test_sortino_translation(self, returns, required_return, translation):
        sr = qrisk.sortino_ratio(returns, required_return)
        sr_depressed = qrisk.sortino_ratio(
            returns-translation,
            required_return-translation)
        sr_raised = qrisk.sortino_ratio(
            returns+translation,
            required_return+translation)
        assert_almost_equal(
            sr,
            sr_depressed,
            DECIMAL_PLACES)
        assert_almost_equal(
            sr,
            sr_raised,
            DECIMAL_PLACES)

#    # Regressive tests for information ratio
#    @parameterized.expand([
#        (empty_returns, 0.0, np.nan),
#        (one_return, 0.0, np.nan),
#        (pos_line, pos_line, np.nan),
#        (mixed_returns, 0.0, 0.10311470414829102),
#        (mixed_returns, simple_benchmark, -0.082491763318632769),
#    ])
#    def test_information_ratio(self, returns, factor_returns, expected):
#        assert_almost_equal(
#            qrisk.information_ratio(returns, factor_returns),
#            expected,
#            DECIMAL_PLACES)

    # The magnitude of the information ratio increases as a higher
    # proportion of returns are uncorrelated with the benchmark.
    @parameterized.expand([
        (flat_line_0, pos_line),
        (flat_line_1, pos_line),
        (noise, pos_line)
    ])
    def test_information_ratio_noisy(self, noise_line, benchmark):
        noisy_returns_1 = noise_line[0:250].add(benchmark[250:], fill_value=0)
        noisy_returns_2 = noise_line[0:500].add(benchmark[500:], fill_value=0)
        noisy_returns_3 = noise_line[0:750].add(benchmark[750:], fill_value=0)
        ir_1 = qrisk.information_ratio(noisy_returns_1, benchmark)
        ir_2 = qrisk.information_ratio(noisy_returns_2, benchmark)
        ir_3 = qrisk.information_ratio(noisy_returns_3, benchmark)
        assert abs(ir_1) < abs(ir_2)
        assert abs(ir_2) < abs(ir_3)

    # Vertical translations change the information ratio in the
    # direction of the translation.
    @parameterized.expand([
        (pos_line, noise, flat_line_1),
        (pos_line, inv_noise, flat_line_1),
        (neg_line, noise, flat_line_1),
        (neg_line, inv_noise, flat_line_1)
    ])
    def test_information_ratio_trans(self, returns, add_noise, translation):
        ir = qrisk.information_ratio(
            returns+add_noise,
            returns
        )
        raised_ir = qrisk.information_ratio(
            returns+add_noise+translation,
            returns
        )
        depressed_ir = qrisk.information_ratio(
            returns+add_noise-translation,
            returns
        )
        assert ir < raised_ir
        assert depressed_ir < ir

#    @parameterized.expand([
#        (empty_returns, simple_benchmark, (np.nan, np.nan)),
#        (one_return, one_return, (np.nan, np.nan)),
#        (mixed_returns, simple_benchmark, (np.nan, np.nan)),
#        (mixed_returns, negative_returns, (-8.3066666666666666,
#                                           -0.71296296296296291)),
#        (mixed_returns, mixed_returns, (0.0, 1.0)),
#        (mixed_returns, -mixed_returns, (0.0, -1.0)),
#    ])
#    def test_alpha_beta(self, returns, benchmark, expected):
#        assert_almost_equal(
#            qrisk.alpha_beta(returns, benchmark)[0],
#            expected[0],
#            DECIMAL_PLACES)
#        assert_almost_equal(
#            qrisk.alpha_beta(returns, benchmark)[1],
#            expected[1],
#            DECIMAL_PLACES)

#    # Regression tests for alpha
#    @parameterized.expand([
#        (empty_returns, simple_benchmark, np.nan),
#        (one_return, one_return, np.nan),
#        (mixed_returns, simple_benchmark, np.nan),
#        (mixed_returns, mixed_returns, 0.0),
#        (mixed_returns, -mixed_returns, 0.0),
#    ])
#    def test_alpha(self, returns, benchmark, expected):
#        assert_almost_equal(
#            qrisk.alpha(returns, benchmark),
#            expected,
#            DECIMAL_PLACES)

    # Alpha/beta translation tests.
    @parameterized.expand([
        (0, .001),
        (0.01, .001),
    ])
    def test_alphabeta_translation(self, mean_returns, translation):
        # Generate correlated returns and benchmark.
        std_returns = 0.01
        correlation = 0.8
        std_bench = .001
        means = [mean_returns, .001]
        covs = [[std_returns**2, std_returns*std_bench*correlation],
                [std_returns*std_bench*correlation, std_bench**2]]
        (ret, bench) = np.random.multivariate_normal(means, covs, 1000).T
        returns = pd.Series(
            ret,
            index=pd.date_range('2000-1-30', periods=1000, freq='D'))
        benchmark = pd.Series(
            bench,
            index=pd.date_range('2000-1-30', periods=1000, freq='D'))
        # Translate returns and generate alphas and betas.
        returns_depressed = returns-translation
        returns_raised = returns+translation
        (alpha_depressed, beta_depressed) = qrisk.alpha_beta(
            returns_depressed, benchmark)
        (alpha_standard, beta_standard) = qrisk.alpha_beta(
            returns, benchmark)
        (alpha_raised, beta_raised) = qrisk.alpha_beta(
            returns_raised, benchmark)
        # Alpha should change proportionally to how much returns were
        # translated.
        assert_almost_equal(
            (alpha_standard - alpha_depressed)/252,
            translation,
            DECIMAL_PLACES)
        assert_almost_equal(
            (alpha_raised - alpha_standard)/252,
            translation,
            DECIMAL_PLACES)
        # Beta remains constant.
        assert_almost_equal(
            beta_standard,
            beta_depressed,
            DECIMAL_PLACES)
        assert_almost_equal(
            beta_standard,
            beta_raised,
            DECIMAL_PLACES)

    # Test alpha/beta with a smaller and larger correlation values.
    @parameterized.expand([
        (0.25, .75),
        (.1, .9),
        (.01, .1)
    ])
    def test_alphabeta_correlation(self, corr_less, corr_more):
        mean_returns = 0.01
        mean_bench = .001
        std_returns = 0.01
        std_bench = .001
        index = pd.date_range('2000-1-30', periods=1000, freq='D')
        # Generate less correlated returns
        means_less = [mean_returns, mean_bench]
        covs_less = [[std_returns**2, std_returns*std_bench*corr_less],
                     [std_returns*std_bench*corr_less, std_bench**2]]
        (ret_less, bench_less) = np.random.multivariate_normal(
            means_less, covs_less, 1000).T
        returns_less = pd.Series(ret_less, index=index)
        benchmark_less = pd.Series(bench_less, index=index)
        # Genereate more highly correlated returns
        means_more = [mean_returns, mean_bench]
        covs_more = [[std_returns**2, std_returns*std_bench*corr_more],
                     [std_returns*std_bench*corr_more, std_bench**2]]
        (ret_more, bench_more) = np.random.multivariate_normal(
            means_more, covs_more, 1000).T
        returns_more = pd.Series(ret_more, index=index)
        benchmark_more = pd.Series(bench_more, index=index)
        # Calculate alpha/beta values
        alpha_less, beta_less = qrisk.alpha_beta(returns_less, benchmark_less)
        alpha_more, beta_more = qrisk.alpha_beta(returns_more, benchmark_more)
        # Alpha determines by how much returns vary from the benchmark return.
        # A lower correlation leads to higher alpha.
        assert alpha_less > alpha_more
        # Beta measures the volatility of returns against benchmark returns.
        # Beta increases proportionally to correlation.
        assert beta_less < beta_more

    @parameterized.expand([
        (empty_returns, simple_benchmark, np.nan),
        (one_return, one_return,  np.nan),
        (mixed_returns, simple_benchmark, np.nan),
        (noise, noise, 1.0),
        (noise, inv_noise, -1.0),
    ])
    def test_beta(self, returns, benchmark, expected):
        assert_almost_equal(
            qrisk.beta(returns, benchmark),
            expected,
            DECIMAL_PLACES)

#    @parameterized.expand([
#        (empty_returns, simple_benchmark),
#        (one_return, one_return),
#        (mixed_returns, simple_benchmark),
#        (mixed_returns, negative_returns),
#        (mixed_returns, mixed_returns),
#        (mixed_returns, -mixed_returns),
#    ])
#    def test_alpha_beta_equality(self, returns, benchmark):
#        alpha_beta = qrisk.alpha_beta(returns, benchmark)
#        assert_almost_equal(
#            alpha_beta[0],
#            qrisk.alpha(returns, benchmark),
#            DECIMAL_PLACES)
#        assert_almost_equal(
#            alpha_beta[1],
#            qrisk.beta(returns, benchmark),
#            DECIMAL_PLACES)

    @parameterized.expand([
        (empty_returns, np.nan),
        (one_return, np.nan),
        (mixed_returns, 0.33072113092134847),
        (simple_benchmark, 1.0),
    ])
    def test_stability_of_timeseries(self, returns, expected):
        assert_almost_equal(
            qrisk.stability_of_timeseries(returns),
            expected,
            DECIMAL_PLACES)

    @parameterized.expand([
        (empty_returns, np.nan),
        (one_return, 1.0),
        (mixed_returns, 0.9473684210526313),
        (np.random.randn(100000), 1.),
    ])
    def test_tail_ratio(self, returns, expected):
        assert_almost_equal(
            qrisk.tail_ratio(returns),
            expected,
            1)
