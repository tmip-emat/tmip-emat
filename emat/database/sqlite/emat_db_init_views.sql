
CREATE INDEX IF NOT EXISTS ema_experiment_measure_run_index
ON ema_experiment_measure(measure_run);





-- Most recent valid run for each experiment
DROP VIEW IF EXISTS ema_experiment_most_recent_valid_run_with_results;
CREATE VIEW ema_experiment_most_recent_valid_run_with_results AS
SELECT
    *,
    max(run_timestamp)
FROM
    ema_experiment_run
WHERE
    run_valid IS NOT FALSE
    AND (
        run_rowid IN (SELECT DISTINCT measure_run FROM ema_experiment_measure)
        OR run_rowid IS NULL
    )
GROUP BY
    experiment_id
;
