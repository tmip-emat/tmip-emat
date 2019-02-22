-- Tables to define emat scope 
DROP TABLE IF EXISTS ema_scope;
DROP TABLE IF EXISTS ema_scope_parameter;
DROP TABLE IF EXISTS ema_scope_measure;

-- ema_scope
CREATE TABLE ema_scope (
    rowid   INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT UNIQUE,
    sheet   TEXT,
    content BLOB
);

CREATE TABLE ema_scope_parameter (
    scope_id      INT NOT NULL,
    parameter_id  INT NOT NULL,
    
    FOREIGN KEY (scope_id) REFERENCES ema_scope(rowid) ON DELETE CASCADE,
    FOREIGN KEY (parameter_id) REFERENCES ema_parameter(rowid)
);


CREATE TABLE ema_scope_measure (
    scope_id      INT NOT NULL,
    measure_id    INT NOT NULL,
    
    FOREIGN KEY (scope_id) REFERENCES ema_scope(rowid) ON DELETE CASCADE,
    FOREIGN KEY (measure_id) REFERENCES ema_measure(rowid)
    
);


CREATE TABLE ema_scope_box (
    box_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_box_id INT,
    scope_id          INT NOT NULL,
    box_name      TEXT UNIQUE,

    FOREIGN KEY (scope_id) REFERENCES ema_scope(rowid) ON DELETE CASCADE
);

CREATE TABLE ema_box_parameter (
    box_id            INT NOT NULL,
    parameter_id      INT NOT NULL,
    threshold_value   NUMERIC,
    threshold_type    INT NOT NULL,

    FOREIGN KEY (box_id) REFERENCES ema_scope_box(box_id) ON DELETE CASCADE,
    FOREIGN KEY (parameter_id) REFERENCES ema_parameter(rowid),
    PRIMARY KEY (box_id, parameter_id, threshold_type)

);

CREATE TABLE ema_box_measure (
    box_id            INT NOT NULL,
    measure_id        INT NOT NULL,
    threshold_value   NUMERIC,
    threshold_type    INT NOT NULL,

    FOREIGN KEY (box_id) REFERENCES ema_scope_box(box_id) ON DELETE CASCADE,
    FOREIGN KEY (measure_id) REFERENCES ema_measure(rowid),
    PRIMARY KEY (box_id, measure_id, threshold_type)
);
