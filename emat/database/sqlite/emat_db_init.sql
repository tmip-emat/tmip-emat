-- Tables to hold designed experiments and the results

CREATE TABLE IF NOT EXISTS ema_tool_info (
    tag        TEXT PRIMARY KEY,
    val
);

CREATE TABLE IF NOT EXISTS ema_log (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    level INTEGER,
    content TEXT,
    Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ema_parameter (
    parameter_id  INTEGER PRIMARY KEY,
    ptype         INTEGER,
    name          TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS ema_measure (
    measure_id     INTEGER PRIMARY KEY,
    name           TEXT UNIQUE,
    transform      TEXT
);

CREATE TABLE IF NOT EXISTS ema_scope (
    scope_id   INTEGER PRIMARY KEY,
    name       TEXT UNIQUE,
    sheet      TEXT,
    content    BLOB
);

CREATE TABLE IF NOT EXISTS ema_scope_parameter (
    scope_id      INT NOT NULL,
    parameter_id  INT NOT NULL,

    FOREIGN KEY (scope_id) REFERENCES ema_scope(scope_id) ON DELETE CASCADE,
    FOREIGN KEY (parameter_id) REFERENCES ema_parameter(parameter_id) ON DELETE CASCADE,
    UNIQUE (scope_id, parameter_id)
);


CREATE TABLE IF NOT EXISTS ema_scope_measure (
    scope_id      INT NOT NULL,
    measure_id    INT NOT NULL,

    FOREIGN KEY (scope_id) REFERENCES ema_scope(scope_id) ON DELETE CASCADE,
    FOREIGN KEY (measure_id) REFERENCES ema_measure(measure_id) ON DELETE CASCADE,
    UNIQUE (scope_id, measure_id)
);


CREATE TABLE IF NOT EXISTS ema_scope_box (
    box_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_box_id INT,
    scope_id          INT NOT NULL,
    box_name      TEXT UNIQUE,

    FOREIGN KEY (scope_id) REFERENCES ema_scope(scope_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ema_box_parameter (
    box_id            INT NOT NULL,
    parameter_id      INT NOT NULL,
    threshold_value   NUMERIC,
    threshold_type    INT NOT NULL,

    FOREIGN KEY (box_id) REFERENCES ema_scope_box(box_id) ON DELETE CASCADE,
    FOREIGN KEY (parameter_id) REFERENCES ema_parameter(parameter_id),
    PRIMARY KEY (box_id, parameter_id, threshold_type)

);

CREATE TABLE IF NOT EXISTS ema_box_measure (
    box_id            INT NOT NULL,
    measure_id        INT NOT NULL,
    threshold_value   NUMERIC,
    threshold_type    INT NOT NULL,

    FOREIGN KEY (box_id) REFERENCES ema_scope_box(box_id) ON DELETE CASCADE,
    FOREIGN KEY (measure_id) REFERENCES ema_measure(measure_id),
    PRIMARY KEY (box_id, measure_id, threshold_type)
);


CREATE TABLE IF NOT EXISTS ema_experiment (
    experiment_id     INTEGER PRIMARY KEY,
    scope_id          INT NOT NULL,

    FOREIGN KEY (scope_id) REFERENCES ema_scope(scope_id)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ema_experiment_run (
    run_rowid        INTEGER PRIMARY KEY,
    run_id           UUID NOT NULL UNIQUE,
    experiment_id    INT NOT NULL,
    run_source       INT NOT NULL DEFAULT 0, -- 0 for core model, or a metamodel_id
    run_status       TEXT,
    run_valid        INT NOT NULL DEFAULT 1,
    run_timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP,
    run_location     TEXT,

    FOREIGN KEY (experiment_id) REFERENCES ema_experiment(experiment_id)
    ON DELETE CASCADE
);



CREATE TABLE IF NOT EXISTS ema_design (
    design_id INTEGER PRIMARY KEY,
    scope_id  INT NOT NULL,
    design    TEXT UNIQUE,

    FOREIGN KEY (scope_id) REFERENCES ema_scope(scope_id)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ema_design_experiment (
    experiment_id      INT NOT NULL,
    design_id          INT NOT NULL,

    FOREIGN KEY (experiment_id) REFERENCES ema_experiment(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (design_id) REFERENCES ema_design(design_id) ON DELETE CASCADE,
    PRIMARY KEY (experiment_id, design_id)
);


CREATE TABLE IF NOT EXISTS ema_experiment_parameter (
    experiment_id      INT NOT NULL,
    parameter_id       INT NOT NULL,
    parameter_value    NUMERIC,
    
    FOREIGN KEY (experiment_id) REFERENCES ema_experiment(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (parameter_id) REFERENCES ema_parameter(parameter_id),
    PRIMARY KEY (experiment_id, parameter_id)
    
);

CREATE TABLE IF NOT EXISTS ema_experiment_measure (
    experiment_id     INT NOT NULL,
    measure_id        INT NOT NULL,
    measure_value     NUMERIC,
    measure_run       INT NOT NULL, -- optional linkage to ema_experiment_run(run_id)

    FOREIGN KEY (experiment_id) REFERENCES ema_experiment(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (measure_id) REFERENCES ema_measure(measure_id) ON DELETE CASCADE,
    FOREIGN KEY (measure_run) REFERENCES ema_experiment_run(run_rowid) ON DELETE CASCADE,
    PRIMARY KEY (experiment_id, measure_id, measure_run)
);

