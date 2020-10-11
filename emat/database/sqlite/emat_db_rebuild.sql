-- Tables to hold designed experiments and the results

PRAGMA foreign_keys = OFF;

CREATE TABLE IF NOT EXISTS xma_parameter (
    parameter_id  INTEGER PRIMARY KEY,
    ptype         INTEGER,
    name          TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS xma_measure (
    measure_id     INTEGER PRIMARY KEY,
    name           TEXT UNIQUE,
    transform      TEXT
);

CREATE TABLE IF NOT EXISTS xma_scope (
    scope_id   INTEGER PRIMARY KEY,
    name       TEXT UNIQUE,
    sheet      TEXT,
    content    BLOB
);

CREATE TABLE IF NOT EXISTS xma_scope_parameter (
    scope_id      INT NOT NULL,
    parameter_id  INT NOT NULL,

    FOREIGN KEY (scope_id) REFERENCES xma_scope(scope_id) ON DELETE CASCADE,
    FOREIGN KEY (parameter_id) REFERENCES xma_parameter(parameter_id) ON DELETE CASCADE,
    UNIQUE (scope_id, parameter_id)
);

CREATE TABLE IF NOT EXISTS xma_scope_measure (
    scope_id      INT NOT NULL,
    measure_id    INT NOT NULL,

    FOREIGN KEY (scope_id) REFERENCES xma_scope(scope_id) ON DELETE CASCADE,
    FOREIGN KEY (measure_id) REFERENCES xma_measure(measure_id) ON DELETE CASCADE,
    UNIQUE (scope_id, measure_id)
);

CREATE TABLE IF NOT EXISTS xma_experiment (
    experiment_id     INTEGER PRIMARY KEY,
    scope_id          INT NOT NULL,

    FOREIGN KEY (scope_id) REFERENCES xma_scope(scope_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS xma_experiment_run (
    run_rowid        INTEGER PRIMARY KEY,
    run_id           UUID NOT NULL UNIQUE,
    experiment_id    INT NOT NULL,
    run_status       TEXT,
    run_valid        INT NOT NULL DEFAULT 1,
    run_timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP,
    run_location     TEXT,
    run_source       INT NOT NULL DEFAULT 0,

    FOREIGN KEY (experiment_id) REFERENCES xma_experiment(experiment_id) ON DELETE CASCADE
);



CREATE TABLE IF NOT EXISTS xma_design (
    design_id INTEGER PRIMARY KEY,
    scope_id  INT NOT NULL,
    design    TEXT UNIQUE,

    FOREIGN KEY (scope_id) REFERENCES xma_scope(scope_id)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS xma_design_experiment (
    experiment_id      INT NOT NULL,
    design_id          INT NOT NULL,

    FOREIGN KEY (experiment_id) REFERENCES xma_experiment(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (design_id) REFERENCES xma_design(design_id) ON DELETE CASCADE,
    PRIMARY KEY (experiment_id, design_id)
);


CREATE TABLE IF NOT EXISTS xma_experiment_parameter (
    experiment_id      INT NOT NULL,
    parameter_id       INT NOT NULL,
    parameter_value    NUMERIC,
    
    FOREIGN KEY (experiment_id) REFERENCES xma_experiment(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (parameter_id) REFERENCES xma_parameter(parameter_id),
    PRIMARY KEY (experiment_id, parameter_id)
);



CREATE TABLE IF NOT EXISTS xma_experiment_measure (
    experiment_id     INT NOT NULL,
    measure_id        INT NOT NULL,
    measure_value     NUMERIC,
    measure_run       INT, -- optional linkage to ema_experiment_run(run_id)

    FOREIGN KEY (experiment_id) REFERENCES xma_experiment(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (measure_id) REFERENCES xma_measure(measure_id),
    FOREIGN KEY (measure_run) REFERENCES xma_experiment_run(run_rowid) ON DELETE CASCADE,
    PRIMARY KEY (experiment_id, measure_id, measure_run)
);


CREATE TABLE IF NOT EXISTS x_meta_model (
   scope_id     INT NOT NULL,
   measure_id       INT NOT NULL,
   lr_r2       NUMERIC,
   gpr_cv      NUMERIC,
   rmse        NUMERIC,

   FOREIGN KEY (scope_id) REFERENCES xma_scope(scope_id) ON DELETE CASCADE,
   FOREIGN KEY (measure_id) REFERENCES xma_measure (measure_id),
   PRIMARY KEY (scope_id, measure_id)
);

CREATE TABLE IF NOT EXISTS x_meta_model_param (
   scope_id     INT NOT NULL,
   measure_id       INT NOT NULL,
   parameter_id      INT NOT NULL,
   est         NUMERIC,
   std_error   NUMERIC,
   pvalue      NUMERIC,

   FOREIGN KEY (scope_id,measure_id) REFERENCES x_meta_model(scope_id,measure_id) ON DELETE CASCADE,
   FOREIGN KEY (parameter_id) REFERENCES xma_parameter(parameter_id),
   PRIMARY KEY (scope_id, measure_id, parameter_id)
);

CREATE TABLE IF NOT EXISTS x_meta_model_pickles (
   scope_id     INT NOT NULL,
   metamodel_id INT NOT NULL,
   name         TEXT,
   pickled_mm   BLOB,

   FOREIGN KEY (scope_id) REFERENCES xma_scope(scope_id) ON DELETE CASCADE,
   PRIMARY KEY (scope_id, metamodel_id)
);


CREATE TABLE IF NOT EXISTS xma_scope_box (
    box_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_box_id INT,
    scope_id          INT NOT NULL,
    box_name      TEXT UNIQUE,

    FOREIGN KEY (scope_id) REFERENCES xma_scope(scope_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS xma_box_parameter (
    box_id            INT NOT NULL,
    parameter_id      INT NOT NULL,
    threshold_value   NUMERIC,
    threshold_type    INT NOT NULL,

    FOREIGN KEY (box_id) REFERENCES xma_scope_box(box_id) ON DELETE CASCADE,
    FOREIGN KEY (parameter_id) REFERENCES xma_parameter(parameter_id),
    PRIMARY KEY (box_id, parameter_id, threshold_type)

);

CREATE TABLE IF NOT EXISTS xma_box_measure (
    box_id            INT NOT NULL,
    measure_id        INT NOT NULL,
    threshold_value   NUMERIC,
    threshold_type    INT NOT NULL,

    FOREIGN KEY (box_id) REFERENCES xma_scope_box(box_id) ON DELETE CASCADE,
    FOREIGN KEY (measure_id) REFERENCES xma_measure(measure_id),
    PRIMARY KEY (box_id, measure_id, threshold_type)
);

INSERT INTO xma_box_measure
SELECT box_id, measure_id, threshold_value, threshold_type FROM ema_box_measure;

INSERT INTO xma_box_parameter
SELECT box_id, parameter_id, threshold_value, threshold_type FROM ema_box_parameter;

INSERT INTO xma_scope_box
SELECT box_id, parent_box_id, scope_id, box_name FROM xma_scope_box;




INSERT INTO x_meta_model_pickles
SELECT scope_id, metamodel_id, name, pickled_mm FROM meta_model_pickles;

INSERT INTO x_meta_model_param
SELECT scope_id, measure_id, parameter_id, est, std_error, pvalue  FROM meta_model_param;

INSERT INTO x_meta_model
SELECT scope_id, measure_id, lr_r2, gpr_cv, rmse FROM meta_model;



INSERT INTO
    xma_experiment_measure
SELECT
    experiment_id  ,
    measure_id     ,
    measure_value  ,
    measure_run
FROM
    ema_experiment_measure;


INSERT INTO
    xma_experiment_parameter
SELECT
    experiment_id, parameter_id, parameter_value
FROM
    ema_experiment_parameter;


INSERT INTO
    xma_design_experiment
SELECT
    experiment_id, design_id
FROM
    ema_design_experiment;


INSERT INTO
    xma_design
SELECT
    rowid,
    scope_id,
    design
FROM
    ema_design;


INSERT INTO
    xma_experiment
SELECT
    rowid,
    scope_id
FROM
    ema_experiment;


INSERT INTO
    xma_scope_measure
SELECT DISTINCT
    scope_id,
    measure_id
FROM
    ema_scope_measure;


INSERT INTO
    xma_scope_parameter
SELECT DISTINCT
    scope_id,
    parameter_id
FROM
    ema_scope_parameter;



INSERT INTO
    xma_scope
SELECT
    rowid,
    name,
    sheet,
    content
FROM
    ema_scope;

INSERT INTO
    xma_measure
SELECT
    rowid,
    name,
    transform
FROM
    ema_measure;



INSERT INTO
    xma_parameter
SELECT
    rowid,
    ptype,
    name
FROM
    ema_parameter;

INSERT INTO
    xma_experiment_run
SELECT
    rowid,
    run_id,
    experiment_id,
    run_status,
    run_valid,
    run_timestamp,
    run_location,
    run_source
FROM
    ema_experiment_run;

DROP VIEW IF EXISTS ema_experiment_with_null_runs;
DROP TABLE IF EXISTS ema_parameter;
DROP TABLE IF EXISTS ema_measure;
DROP TABLE IF EXISTS ema_scope;
DROP TABLE IF EXISTS ema_scope_parameter;
DROP TABLE IF EXISTS ema_scope_measure;
DROP TABLE IF EXISTS ema_experiment;
DROP TABLE IF EXISTS ema_design;
DROP TABLE IF EXISTS ema_design_experiment;
DROP TABLE IF EXISTS ema_experiment_parameter;
DROP TABLE IF EXISTS ema_experiment_measure;
DROP TABLE IF EXISTS ema_experiment_run;
DROP TABLE IF EXISTS ema_box_measure  ;
DROP TABLE IF EXISTS ema_box_parameter;
DROP TABLE IF EXISTS ema_scope_box    ;
DROP TABLE IF EXISTS ema_duplicate_experiment;

DROP TABLE IF EXISTS meta_model_pickles;
DROP TABLE IF EXISTS meta_model_param;
DROP TABLE IF EXISTS meta_model;


DROP INDEX IF EXISTS ema_experiment_measure_run_index;
DROP VIEW IF EXISTS ema_experiment_most_recent_valid_run_with_results;

ALTER TABLE xma_parameter            RENAME TO ema_parameter           ;
ALTER TABLE xma_measure              RENAME TO ema_measure             ;
ALTER TABLE xma_scope                RENAME TO ema_scope               ;
ALTER TABLE xma_scope_parameter      RENAME TO ema_scope_parameter     ;
ALTER TABLE xma_scope_measure        RENAME TO ema_scope_measure       ;
ALTER TABLE xma_experiment           RENAME TO ema_experiment          ;
ALTER TABLE xma_design               RENAME TO ema_design              ;
ALTER TABLE xma_design_experiment    RENAME TO ema_design_experiment   ;
ALTER TABLE xma_experiment_parameter RENAME TO ema_experiment_parameter;
ALTER TABLE xma_experiment_measure   RENAME TO ema_experiment_measure  ;
ALTER TABLE xma_experiment_run       RENAME TO ema_experiment_run      ;
ALTER TABLE xma_box_measure          RENAME TO ema_box_measure         ;
ALTER TABLE xma_box_parameter        RENAME TO ema_box_parameter       ;
ALTER TABLE xma_scope_box            RENAME TO ema_scope_box           ;
ALTER TABLE x_meta_model_pickles     RENAME TO meta_model_pickles      ;
ALTER TABLE x_meta_model_param       RENAME TO meta_model_param        ;
ALTER TABLE x_meta_model             RENAME TO meta_model              ;


CREATE INDEX IF NOT EXISTS ema_experiment_measure_run_index
ON ema_experiment_measure(measure_run);

-- Most recent valid run for each experiment
PRAGMA foreign_key_check;

PRAGMA foreign_keys = ON