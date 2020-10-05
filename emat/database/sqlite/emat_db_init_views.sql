
CREATE INDEX IF NOT EXISTS ema_experiment_measure_run_index
ON ema_experiment_measure(measure_run);



DROP VIEW IF EXISTS ema_experiment_with_null_runs;
CREATE VIEW IF NOT EXISTS ema_experiment_with_null_runs AS
SELECT DISTINCT
    null AS run_rowid,
    null AS run_id,
    experiment_id,
    null AS run_status,
    true AS run_valid,
    0 AS run_timestamp,
    null AS run_location,
    0 AS run_source
FROM
    ema_experiment
UNION
SELECT * FROM ema_experiment_run
;



-- Most recent valid run for each experiment
DROP VIEW IF EXISTS ema_experiment_most_recent_valid_run_with_results;
CREATE VIEW ema_experiment_most_recent_valid_run_with_results AS
SELECT
    *,
    max(run_timestamp)
FROM
    ema_experiment_with_null_runs
WHERE
    run_valid IS NOT FALSE
    AND (
        run_rowid IN (SELECT DISTINCT measure_run FROM ema_experiment_measure)
        OR run_rowid IS NULL
    )
GROUP BY
    experiment_id
;
