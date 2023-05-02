#!/usr/bin/env python3
from random import random, seed
from typing import Optional, Tuple, List
import time
from collections import deque
from sortedcontainers import SortedDict
from scipy.stats import binom

"""
These are the algorithms used to correct a raw score for guessing on a multiple choice
test that uses a ceiling, such as an IQ test. The procedure and context is given in:

Basham, R.B. & St. Jules, K.E. (2023). An Evaluation of the Effectiveness of the Second 
    Chance Protocol for Reducing Underperformance During Intellectual Testing", under review.
    
Run this file or call test_guessing() to calculate the distribution used for Table 2 in the paper.
Requires Python 3.5 due to the typing module. If type hints are removed, should run under earlier
versions of Python 3.

If the algorithm is running correctly, test_guessing() will return something close to the 
following table with mean raw score = 1.38 +/- .01

Frequency distribution of estimated raw score increases due to FPs, 
   do_adjust_prob = False, for 100,000 trials:
     0:     27.63%      27.63%    100.00%
     1:     31.77%      59.40%     72.37%
     2:     22.36%      81.76%     40.60%
     3:     12.02%      93.78%     18.24%
     4:      4.74%      98.52%      6.22%
     5:      1.25%      99.77%      1.48%
     6:      0.20%      99.97%      0.23%
     7:      0.03%     100.00%      0.03%
     8:      0.00%     100.00%      0.00%
Mean raw score increase due to FPs = 1.392

"""


def build_second_chance_distro(sa_responses: str,
                               ceiling_fails: int,
                               ceiling_n: int,
                               n_response_options: int,
                               scale_length: int,
                               do_adjust_prob: Optional[bool] = True,
                               n_trials: Optional[int] = int(1e5)) -> Tuple[float, SortedDict]:
    """Return the mean and distribution of False Positives (FP) for a multiple-choice test
    during Second Chance (SC) administration.

    The test has a ceiling of 'ceiling_fails' failures out of 'ceiling_n' items,
    'n_response_options', and a scale length if 'scale_length', not all of which need to be
    administered. 'sa_responses' is a string of subject responses during standard administration,
    with '0' indicating a fail, '1' a pass. Set ceiling_n to None for a test that does not use
    ceiling-termination.

    The returned distribution of False Positives is a SortedDict structured as
    {n_false_positives: frequency}

    The returned likelihood distribution can be used to determine the likelihood of obtained a
    specified score increase during SC administration due to guessing, assuming the subject does
    not know the answers to any items failed during Standard Administration (SA), nor to any of
    the new items given during SC.
    """
    scores_dist = SortedDict()
    item_score_increases = 0

    for trial in range(n_trials):
        score_increase = compute_guessing_trial(sa_responses, ceiling_fails, ceiling_n,
                                                n_response_options, scale_length, do_adjust_prob)
        item_score_increases += score_increase
        if score_increase in scores_dist:
            scores_dist[score_increase] = scores_dist[score_increase] + 1
        else:
            scores_dist[score_increase] = 1

    return item_score_increases/n_trials, scores_dist


def compute_guessing_trial(sa_items: str,
                           ceiling_fails: int,
                           ceiling_n: int,
                           n_response_options: int,
                           scale_length: int,
                           do_adjust_prob: Optional[bool] = False) -> int:
    """Return one randomized trial given the number of false positives (FPs) during second
    chance administration for given parameters. Cumulate these trials to obtain an estimate
    of the population distribution.

    'sa_items' is a string of all Standard Administration (SA) responses, coded as '0' = fail and
    '1' = pass. Testing is discontinued when the subject fails 'ceiling_fails' items out of
    'ceiling_n' items, or has been given all 'scale-length' items. do_adust_prob indicates
    whether to increase the base-rate during SC, on the assumption that the subject recalls
    their response during SA and never gives that as their guess.

    This assumes that second chance was administered only for items missed during SA. Items
    beyond the SA items will be given until a SC ceiling is established, or the end of the scale
    is reached.

    PERFORMANCE NOTE: The is not optimized for readability, not performance. Calls to
    last_items_deq.append(item_score) and sum(last_items_deq) account for almost 50% of
    the execution time. Using a list instead of a deque doesn't help. Typical execution time
    is around 300 - 500 ms for 100K calls.
    """
    n_fps, last_items_deq, sa_length = 0, deque(maxlen=ceiling_n), len(sa_items)
    start_off = sa_items.find('0')  # Offset of first SA fail, first SC item
    sa_prob = 1 / n_response_options
    sc_prob = 1 / (n_response_options - 1) if do_adjust_prob else sa_prob
    for off in range(start_off, scale_length - 1):
        if off < sa_length:  # Is this an item given during SA?
            if sa_items[off] == '0':  # Was item failed during SA?
                item_score = int(random() <= sc_prob)
                n_fps += item_score
            else:
                item_score = 1
        else:  # This item was not given during SA.
            item_score = int(random() <= sa_prob)
            n_fps += item_score
        last_items_deq.append(item_score)
        if len(last_items_deq) == ceiling_n:
            if sum(last_items_deq) == ceiling_n - ceiling_fails:  # Reached ceiling?
                break
    return n_fps


def calc_likelihood(raw_score_increase: int,
                    dist: SortedDict) -> Tuple[float, float, float]:
    """Return the mid_value,lower_bound, and upper_bound of the likelihood of the obtained
    raw_score_increase being due to random guessing.

    NOTE: 'dist' is treated as a population likelihood distribution, so it should have a large N,
    such as 10K or more.
    """
    n_less_than, n_greater_than, n = 0, 0, 0
    for increase, freq in dist.items():
        if increase < raw_score_increase:
            n_less_than += freq
        elif increase > raw_score_increase:
            n_greater_than += freq
        n += freq
    upper = 1 - n_less_than / n
    lower = n_greater_than / n
    mid = (lower + upper) / 2
    return mid, lower, upper


def calc_second_chance_stats(sa_responses: str,
                             scale_length: int,
                             raw_score_increase: int,
                             n_readmin_fail: int,
                             n_readmin_same: int,
                             significance_level: float,
                             n_response_options: int,
                             ceiling_n: Optional[int],
                             ceiling_fails: Optional[int],
                             n_trials: Optional[int] = int(1e4),
                             do_print_table: Optional[bool] = False) -> \
        Tuple[float, float, Optional[SortedDict]]:
    """Return the distribution mean, the likelihood of the number of correct answers during
    Second Chance testing, and the full distribution of false positives (a SortedDict
    structured as {n_false_positives: frequency}).

    sa_response is a string of responses during standard administration,
    where '2' = pass, '1' = fail.
    """

    final_responses = sa_responses.replace('1', '0')
    final_responses = final_responses.replace('2', '1')

    # base_rate is the likelihood of giving the same wrong answer during Second Chance (SC).
    # Assume there are 6 response options, A, B, C, D, E, and F, with A being the correct
    # answer. There are only 5 ways to fail an item during SC, by responding B, C, D, E, or F,
    # so the odds of giving the same wrong answer during SC is 1/5, not 1/6.

    base_rate = 1 / (n_response_options - 1)

    # gte_same_binom_prob is the probability of giving at least n_readmin_same same wrong answers
    # during Second Chance, out of n_readmin_fails failed items.

    if n_readmin_same:
        # binom.cdf() params are k, n, p.
        gte_same_binom_prob = 1 - float(binom.cdf(n_readmin_same-1, n_readmin_fail, base_rate))
    else:
        gte_same_binom_prob = 1.0

    if gte_same_binom_prob < significance_level:
        # Set n_fp to zero because subject does not appear to have guessed.
        return 0.0, 1.0, None
    else:
        lte_same_binom_prob = float(binom.cdf(n_readmin_same, n_readmin_fail, base_rate))
        # Test to see if subject had less of the same wrong answers during Second Chance than
        # would be likely when guessing. This is an indication that the subject remembered their
        # answers from standard administration and deliberately did not give the same answer
        # during Second Chance. The power of this statistical test is often quite weak.
        if lte_same_binom_prob <= significance_level:
            do_adjust_prob = True
        else:
            do_adjust_prob = False

    start_pc = time.perf_counter()
    mean, dist = build_second_chance_distro(sa_responses=final_responses,
                                            ceiling_fails=ceiling_fails,
                                            ceiling_n=ceiling_n,
                                            n_response_options=n_response_options,
                                            scale_length=scale_length,
                                            n_trials=n_trials,
                                            do_adjust_prob=do_adjust_prob)

    mid_prob, lower_prob, upper_prob = calc_likelihood(raw_score_increase, dist)
    dur = time.perf_counter() - start_pc

    if do_print_table:
        print(f'\nFrequency distribution of estimated raw score increases due to FPs, \n   '
              f'do_adjust_prob = {do_adjust_prob}, for {n_trials:,} trials:')
        cum_freq, prior_freq = 0, 0
        for n, count in dist.items():
            cum_freq += count
            print(f'   {n:>3}: {count/n_trials:>10.2%}  {cum_freq/n_trials:>10.2%} '
                  f'{1 - prior_freq/n_trials:>10.2%}')
            prior_freq += count
        print(f'Mean raw score increase due to FPs = {mean:.3f}')
        print(f'\nExecution took {dur*1000:.3f} ms.')

    return mean, upper_prob, dist


def compute_sc_value_ranges(n_trials: int = 100):
    """Repeatedly call test_guessing() to determine the min and max frequencies for
    each number of FPs for n_trials."""

    smallest, greatest = SortedDict(), SortedDict()
    start_pc = time.perf_counter()
    for x in range(n_trials):
        result_tuple = test_guessing(random_seed=1,
                                     do_print_table=False)
        mean, prob, dist = result_tuple

        for key, val in dist.items():
            # debug1 = smallest[key] if key in smallest else None
            if key not in smallest or val < smallest[key]:
                smallest[key] = val
            if key not in greatest or val > greatest[key]:
                greatest[key] = val
    dur = time.perf_counter() - start_pc
    text = f'\n\ncompute_sc_value_ranges() completed {n_trials} trials in {dur:.3f} seconds' \
           f' with following results:'
    print(text)
    text = f'   Smallest = {smallest}\n   Largest =  {greatest}'
    print(text)


def binom_pdf_lookup(k, n, p):
    """Return the likelihood of having k successes out of n trials with p likelihood of
    success per trial.
    """
    binom_prob = float(binom.pmf(k, n, p))
    print(f'The binomial probability of {k} successes out of {n} trials with p = '
          f'{p:.3f} is {binom_prob:.6f}')
    return


def binom_cdf_lookup(k, n, p):
    """Return the cumulative likelihood of getting k or fewer successes out of n
    trials with p likelihood of success per trial. The likelihood of getting k or more
    successes is given by 1 - binom.cdf(k-1, n, p), if k > 0.
    """
    binom_prob = float(binom.cdf(k, n, p))
    print(f'The cumulative binomial probability of {k} successes out of {n} trials with '
          f'p = {p:.3f} is {binom_prob:.6f}')
    return


def test_guessing(random_seed: Optional[int] = 1,
                  do_print_table: Optional[bool] = True,
                  do_verify: Optional[bool] = False):
    """Recalculate the distribution used in Figure 1 of the paper.

    If do_verify = True, sets the random_seed to 1 and verifies that it returns the correct
    frequencies for parameters given below under "Data from Figure 1".

    If random_seed is not set, the algorithm uses random variables and result will vary
    with each run, though the calculated mean should be within +/- .01 of the value given
    in the paper when using 100,000 trials (1e5)
    """

    if do_verify:
        seed(1)
    elif random_seed:
        seed(random_seed)  # Seed the random function to get same results each time.

    # Data from Figure 1:
    sa_responses = '22222222222222222222222222222222212112111'
    sc_responses = '.................................2.21.12111211'
    sc_raw_score = sc_responses.count('2')
    scale_length = 46
    n_trials = int(1e5)  # 100,000 trials is usually sufficient.
    n_response_options = 6  # This would be 2 for a True/False test
    significance_level = .15  # .15 is probably more appropriate than .05 for case-by-case test
    n_readmin_fail = 7  # N of re-administered items that were failed
    n_readmin_same = 0  # N of re-administered items where the subject gave same wrong answer
    ceiling_n = 4  # Set this to None for a test without ceiling termination.
    ceiling_fails = 4  # Set this to None for a test without ceiling termination.

    result_tuple = calc_second_chance_stats(sa_responses=sa_responses,
                                            scale_length=scale_length,
                                            raw_score_increase=sc_raw_score,
                                            n_readmin_fail=n_readmin_fail,
                                            n_readmin_same=n_readmin_same,
                                            significance_level=significance_level,
                                            n_response_options=n_response_options,
                                            ceiling_n=ceiling_n,
                                            ceiling_fails=ceiling_fails,
                                            n_trials=n_trials,
                                            do_print_table=do_print_table)
    mean, prob, dist = result_tuple
    if do_verify:
        expected_freqs = [27633, 31771, 22360, 12020, 4735, 1249, 204, 25, 3]
        off = 0
        for freq in dist.values():
            if freq != expected_freqs[off]:
                text = f'Bad frequency for offset {off}: {freq}'
                print(text)
                return
            off += 1
        text = f'Successfully passed the validation test.'
        print(text)

    return result_tuple


if __name__ == '__main__':
    binom_pdf_lookup(0, 5, 1/5)
    # binom_pdf_lookup(0, 6, 1/5)
    # binom_pdf_lookup(0, 7, 1/5)
    # binom_pdf_lookup(0, 8, 1/5)
    # binom_pdf_lookup(0, 9, 1/5)
    # print(f'\n')
    binom_cdf_lookup(0, 6, 1/5)
    # binom_cdf_lookup(1, 6, 1/5)
    # binom_cdf_lookup(2, 6, 1/5)
    # binom_cdf_lookup(3, 6, 1/5)
    # binom_cdf_lookup(4, 6, 1/5)
    # binom_cdf_lookup(5, 6, 1/5)
    # binom_cdf_lookup(6, 6, 1/5)

    test_guessing(do_verify=True)


