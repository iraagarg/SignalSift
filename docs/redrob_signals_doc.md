# Redrob Behavioral Signals — Reference

The 23 signals in each candidate's `redrob_signals` object represent simulated platform activity/engagement. They are often more predictive of whether a candidate can actually be hired than their static profile — a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is, for hiring purposes, not actually available. Use these as a multiplier/modifier on top of skill-match scoring, not as an independent score.

| # | Signal | Range/type | What it measures |
|---|---|---|---|
| 1 | profile_completeness_score | 0-100 | How much of the profile is filled in |
| 2 | signup_date | date | When they signed up on Redrob |
| 3 | last_active_date | date | When they last logged in |
| 4 | open_to_work_flag | bool | Marked themselves available |
| 5 | profile_views_received_30d | int ≥0 | Recruiter profile views, last 30d |
| 6 | applications_submitted_30d | int ≥0 | Roles applied to recently |
| 7 | recruiter_response_rate | 0.0-1.0 | Fraction of recruiter messages replied to |
| 8 | avg_response_time_hours | ≥0 | Median time to respond to a recruiter |
| 9 | skill_assessment_scores | dict[skill→0-100] | Per-skill Redrob assessment scores |
| 10 | connection_count | int ≥0 | Redrob connections |
| 11 | endorsements_received | int ≥0 | Total skill endorsements |
| 12 | notice_period_days | 0-180 | Stated notice period |
| 13 | expected_salary_range_inr_lpa.min/.max | ≥0 | Salary expectation, INR LPA |
| 14 | preferred_work_mode | onsite/hybrid/remote/flexible | Work-mode preference |
| 15 | willing_to_relocate | bool | Will relocate if needed |
| 16 | github_activity_score | -1 to 100 | GitHub activity (-1 = no GitHub linked) |
| 17 | search_appearance_30d | int ≥0 | Times appeared in recruiter searches, 30d |
| 18 | saved_by_recruiters_30d | int ≥0 | Recruiter bookmarks, 30d |
| 19 | interview_completion_rate | 0.0-1.0 | Fraction of scheduled interviews attended |
| 20 | offer_acceptance_rate | -1 to 1.0 | Historical offer acceptance (-1 = no offer history) |
| 21 | verified_email | bool | Email verified |
| 22 | verified_phone | bool | Phone verified |
| 23 | linkedin_connected | bool | LinkedIn connected |
