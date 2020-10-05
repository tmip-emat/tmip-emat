-- Meta Model

CREATE TABLE IF NOT EXISTS meta_model (
   scope_id     INT NOT NULL,
   measure_id       INT NOT NULL,
   lr_r2       NUMERIC,
   gpr_cv      NUMERIC,
   rmse        NUMERIC,
	
   FOREIGN KEY (scope_id) REFERENCES ema_scope(scope_id) ON DELETE CASCADE,
   FOREIGN KEY (measure_id) REFERENCES ema_measure (measure_id),
   PRIMARY KEY (scope_id, measure_id)
);

CREATE TABLE IF NOT EXISTS meta_model_param (
   scope_id     INT NOT NULL,
   measure_id       INT NOT NULL,
   parameter_id      INT NOT NULL,
   est         NUMERIC,
   std_error   NUMERIC,
   pvalue      NUMERIC,
	
   FOREIGN KEY (scope_id,measure_id) REFERENCES meta_model(scope_id,measure_id) ON DELETE CASCADE,
   FOREIGN KEY (parameter_id) REFERENCES ema_parameter(parameter_id),
   PRIMARY KEY (scope_id, measure_id, parameter_id)
);

CREATE TABLE IF NOT EXISTS meta_model_pickles (
   scope_id     INT NOT NULL,
   metamodel_id INT NOT NULL,
   name         TEXT,
   pickled_mm   BLOB,

   FOREIGN KEY (scope_id) REFERENCES ema_scope(scope_id) ON DELETE CASCADE,
   PRIMARY KEY (scope_id, metamodel_id)
);
