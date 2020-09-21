# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 09:41:27 2018

@author: mmilkovits
"""


CONDITIONAL_INSERT_XL = (
    '''INSERT OR IGNORE INTO ema_parameter( name, ptype )
        VALUES(?1, CASE WHEN ?2 LIKE '%uncertainty%' THEN 1 WHEN ?2 LIKE '%constant%' THEN 2 ELSE 0 END)
    '''
    )

CONDITIONAL_INSERT_M = (
    '''INSERT OR IGNORE INTO ema_measure( name, transform )
        VALUES(?,?)
    '''
    )

INSERT_SCOPE = (
    '''INSERT INTO ema_scope( name, sheet, content )
        VALUES(?1, ?2, ?3)
    '''
    )

GET_SCOPE = (
    '''SELECT content FROM ema_scope WHERE name = ?'''
)

DELETE_SCOPE = (
    ''' DELETE from ema_scope WHERE name = ?
    '''
    )

INSERT_SCOPE_XL = (
    '''INSERT INTO ema_scope_parameter( scope_id, parameter_id )
        SELECT ema_scope.rowid, ema_parameter.rowid FROM
            ema_scope JOIN ema_parameter
            WHERE ema_scope.name = ? AND ema_parameter.name = ?
    '''
    )

INSERT_SCOPE_M = (
    '''INSERT INTO ema_scope_measure( scope_id, measure_id )
        SELECT ema_scope.rowid, ema_measure.rowid FROM
            ema_scope JOIN ema_measure
            WHERE ema_scope.name = ? AND ema_measure.name = ?
    '''
    )

GET_SCOPE_XL = (
    '''SELECT ema_parameter.name
        FROM ema_parameter JOIN ema_scope_parameter sv ON (ema_parameter.rowid = sv.parameter_id)
        JOIN ema_scope s ON (sv.scope_id = s.rowid)
        WHERE s.name = ?
    '''
    )

GET_SCOPE_X = (
    '''SELECT ema_parameter.name
        FROM ema_parameter JOIN ema_scope_parameter sv ON (ema_parameter.rowid = sv.parameter_id)
        JOIN ema_scope s ON (sv.scope_id = s.rowid)
        WHERE s.name = ?
        AND ema_parameter.ptype = 1
    '''
    )

GET_SCOPE_L = (
    '''SELECT ema_parameter.name
        FROM ema_parameter JOIN ema_scope_parameter sv ON (ema_parameter.rowid = sv.parameter_id)
        JOIN ema_scope s ON (sv.scope_id = s.rowid)
        WHERE s.name = ?
        AND ema_parameter.ptype = 0
    '''
    )

GET_SCOPE_C = (
    '''SELECT ema_parameter.name
        FROM ema_parameter JOIN ema_scope_parameter sv ON (ema_parameter.rowid = sv.parameter_id)
        JOIN ema_scope s ON (sv.scope_id = s.rowid)
        WHERE s.name = ?
        AND ema_parameter.ptype = 2
    '''
    )

GET_SCOPE_M = (
    '''SELECT ema_measure.name
        FROM ema_measure JOIN ema_scope_measure sp ON (ema_measure.rowid = sp.measure_id)
        JOIN ema_scope s ON (sp.scope_id = s.rowid)
        WHERE s.name = ?
    '''
    )

INSERT_EX = (
    '''INSERT INTO ema_experiment ( scope_id, design )
            SELECT ema_scope.rowid, ?
            FROM ema_scope WHERE ema_scope.name = ?
    '''
    )

INSERT_EX_DUPLICATE = (
    '''INSERT INTO ema_duplicate_experiment ( scope_id, design, orig_id )
            SELECT ema_scope.rowid, ?1, ?3
            FROM ema_scope WHERE ema_scope.name = ?2
    '''
    )


DELETE_EX = (
    '''DELETE FROM ema_experiment
       WHERE rowid IN (
        SELECT ema_experiment.rowid
            FROM ema_experiment JOIN ema_scope s ON (ema_experiment.scope_id = s.rowid)
            WHERE s.name = ? AND ema_experiment.design = ?
        UNION
        SELECT ema_duplicate_experiment.orig_id
            FROM ema_duplicate_experiment JOIN ema_scope s ON (ema_duplicate_experiment.scope_id = s.rowid)
            WHERE s.name = ? AND ema_duplicate_experiment.design = ?
       )
    '''
    )

INSERT_EX_XL = (
    '''INSERT INTO ema_experiment_parameter( experiment_id, parameter_id, parameter_value )
        SELECT ?, ema_parameter.rowid, ? FROM
            ema_parameter WHERE ema_parameter.name = ?
    '''
    )

GET_EX_XL = (
    '''SELECT experiment_id, ema_parameter.name, parameter_value
            FROM ema_experiment_parameter JOIN ema_parameter on ema_experiment_parameter.parameter_id = ema_parameter.rowid
            JOIN ema_experiment ON ema_experiment_parameter.experiment_id = ema_experiment.rowid
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE s.name =?1 and ema_experiment.design = ?2
        UNION ALL
        SELECT experiment_id, ema_parameter.name, parameter_value
            FROM ema_experiment_parameter JOIN ema_parameter on ema_experiment_parameter.parameter_id = ema_parameter.rowid
            JOIN ema_duplicate_experiment ON ema_experiment_parameter.experiment_id = ema_duplicate_experiment.orig_id
            JOIN ema_scope s on ema_duplicate_experiment.scope_id = s.rowid
            WHERE s.name =?1 and ema_duplicate_experiment.design = ?2;
    '''
    )

GET_EXPERIMENT_IDS_BY_VALUE = (
    '''SELECT experiment_id
            FROM ema_experiment_parameter JOIN ema_parameter on ema_experiment_parameter.parameter_id = ema_parameter.rowid
            JOIN ema_experiment ON ema_experiment_parameter.experiment_id = ema_experiment.rowid
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE s.name =?1 and ema_parameter.name = ?2 and parameter_value = ?3;
    '''
    )

GET_EXPERIMENT_IDS_BY_DESIGN_AND_VALUE = (
    '''SELECT experiment_id
            FROM ema_experiment_parameter JOIN ema_parameter on ema_experiment_parameter.parameter_id = ema_parameter.rowid
            JOIN ema_experiment ON ema_experiment_parameter.experiment_id = ema_experiment.rowid
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE s.name =?1 and ema_experiment.design = ?2 and ema_parameter.name = ?3 and parameter_value = ?4;
    '''
    )


GET_EX_XL_ALL = (
    '''SELECT experiment_id, ema_parameter.name, parameter_value
            FROM ema_experiment_parameter JOIN ema_parameter on ema_experiment_parameter.parameter_id = ema_parameter.rowid
            JOIN ema_experiment ON ema_experiment_parameter.experiment_id = ema_experiment.rowid
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE s.name =?1;
    '''
    )

GET_EX_XL_IDS_IN = (
    '''SELECT experiment_id, ema_parameter.name, parameter_value
            FROM ema_experiment_parameter JOIN ema_parameter on ema_experiment_parameter.parameter_id = ema_parameter.rowid
            JOIN ema_experiment ON ema_experiment_parameter.experiment_id = ema_experiment.rowid
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE s.name =?1 AND experiment_id in (???);
    '''
    )

GET_EX_XL_ALL_PENDING = (
    '''SELECT experiment_id, ema_parameter.name, parameter_value
            FROM ema_experiment_parameter 
            JOIN ema_parameter ON ema_experiment_parameter.parameter_id = ema_parameter.rowid
            JOIN ema_experiment ON ema_experiment_parameter.experiment_id = ema_experiment.rowid
            JOIN ema_scope s ON ema_experiment.scope_id = s.rowid
            WHERE s.name =?1
            AND experiment_id NOT IN (
                SELECT ema_experiment_measure.experiment_id
                FROM ema_experiment_measure 
                JOIN ema_measure on ema_experiment_measure.measure_id = ema_measure.rowid
                JOIN ema_experiment ON ema_experiment_measure.experiment_id = ema_experiment.rowid
                JOIN ema_scope s on ema_experiment.scope_id = s.rowid
                WHERE s.name =?1
            )
    '''
    )


GET_EX_XL_PENDING = (
    '''SELECT experiment_id, ema_parameter.name, parameter_value
            FROM ema_experiment_parameter 
            JOIN ema_parameter ON ema_experiment_parameter.parameter_id = ema_parameter.rowid
            JOIN ema_experiment ON ema_experiment_parameter.experiment_id = ema_experiment.rowid
            JOIN ema_scope s ON ema_experiment.scope_id = s.rowid
            WHERE s.name =?1 AND ema_experiment.design = ?2
            AND experiment_id NOT IN (
                SELECT ema_experiment_measure.experiment_id
                FROM ema_experiment_measure 
                JOIN ema_measure on ema_experiment_measure.measure_id = ema_measure.rowid
                JOIN ema_experiment ON ema_experiment_measure.experiment_id = ema_experiment.rowid
                JOIN ema_scope s on ema_experiment.scope_id = s.rowid
                WHERE s.name =?1 AND ema_experiment.design = ?2
            )
        UNION ALL
        SELECT experiment_id, ema_parameter.name, parameter_value
            FROM ema_experiment_parameter 
            JOIN ema_parameter ON ema_experiment_parameter.parameter_id = ema_parameter.rowid
            JOIN ema_duplicate_experiment ON ema_experiment_parameter.experiment_id = ema_duplicate_experiment.orig_id
            JOIN ema_scope s ON ema_duplicate_experiment.scope_id = s.rowid
            WHERE s.name =?1 AND ema_duplicate_experiment.design = ?2
            AND experiment_id NOT IN (
                SELECT ema_experiment_measure.experiment_id
                FROM ema_experiment_measure 
                JOIN ema_measure on ema_experiment_measure.measure_id = ema_measure.rowid
                JOIN ema_duplicate_experiment ON ema_experiment_measure.experiment_id = ema_duplicate_experiment.orig_id
                JOIN ema_scope s on ema_duplicate_experiment.scope_id = s.rowid
                WHERE s.name =?1 AND ema_duplicate_experiment.design = ?2
            )
    '''
    )

INSERT_EX_M = (
    '''INSERT OR REPLACE INTO ema_experiment_measure( experiment_id, measure_id, measure_value, measure_source )
        SELECT ?, ema_measure.rowid, ?, ? FROM
            ema_measure WHERE ema_measure.name = ?
    '''
    )

GET_EX_XLM = (
    '''
    SELECT experiment_id, ema_parameter.name, parameter_value
            FROM ema_parameter JOIN ema_experiment_parameter on ema_experiment_parameter.parameter_id = ema_parameter.rowid
            JOIN ema_experiment ON ema_experiment_parameter.experiment_id = ema_experiment.rowid
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE s.name =?1 and ema_experiment.design = ?2
    UNION
    SELECT experiment_id, ema_measure.name, measure_value
            FROM ema_experiment_measure JOIN ema_measure on ema_experiment_measure.measure_id = ema_measure.rowid
            JOIN ema_experiment ON ema_experiment_measure.experiment_id = ema_experiment.rowid
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE s.name =?1 and ema_experiment.design = ?2
            /*source*/
    UNION
    SELECT experiment_id, ema_parameter.name, parameter_value
            FROM ema_parameter JOIN ema_experiment_parameter on ema_experiment_parameter.parameter_id = ema_parameter.rowid
            JOIN ema_duplicate_experiment ON ema_experiment_parameter.experiment_id = ema_duplicate_experiment.orig_id
            JOIN ema_scope s on ema_duplicate_experiment.scope_id = s.rowid
            WHERE s.name =?1 and ema_duplicate_experiment.design = ?2
    UNION
    SELECT experiment_id, ema_measure.name, measure_value
            FROM ema_experiment_measure JOIN ema_measure on ema_experiment_measure.measure_id = ema_measure.rowid
            JOIN ema_duplicate_experiment ON ema_experiment_measure.experiment_id = ema_duplicate_experiment.orig_id
            JOIN ema_scope s on ema_duplicate_experiment.scope_id = s.rowid
            WHERE s.name =?1 and ema_duplicate_experiment.design = ?2
            /*source*/
    '''
    )

GET_EX_XLM_BYSOURCE = GET_EX_XLM.replace('/*source*/', ' AND ema_experiment_measure.measure_source =?3')


GET_EX_M = (
    '''
    SELECT experiment_id, ema_measure.name, measure_value
            FROM ema_experiment_measure JOIN ema_measure on ema_experiment_measure.measure_id = ema_measure.rowid
            JOIN ema_experiment ON ema_experiment_measure.experiment_id = ema_experiment.rowid
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE s.name =?1 and ema_experiment.design = ?2
            /*source*/
    UNION
    SELECT experiment_id, ema_measure.name, measure_value
            FROM ema_experiment_measure JOIN ema_measure on ema_experiment_measure.measure_id = ema_measure.rowid
            JOIN ema_duplicate_experiment ON ema_experiment_measure.experiment_id = ema_duplicate_experiment.orig_id
            JOIN ema_scope s on ema_duplicate_experiment.scope_id = s.rowid
            WHERE s.name =?1 and ema_duplicate_experiment.design = ?2
            /*source*/
    '''
    )

GET_EX_M_BY_ID = (
    '''
    SELECT experiment_id, ema_measure.name, measure_value
            FROM ema_experiment_measure JOIN ema_measure on ema_experiment_measure.measure_id = ema_measure.rowid
            JOIN ema_experiment ON ema_experiment_measure.experiment_id = ema_experiment.rowid
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE s.name =?1 and ema_experiment.design = ?2 and experiment_id = ?3
            /*source*/
    UNION
    SELECT experiment_id, ema_measure.name, measure_value
            FROM ema_experiment_measure JOIN ema_measure on ema_experiment_measure.measure_id = ema_measure.rowid
            JOIN ema_duplicate_experiment ON ema_experiment_measure.experiment_id = ema_duplicate_experiment.orig_id
            JOIN ema_scope s on ema_duplicate_experiment.scope_id = s.rowid
            WHERE s.name =?1 and ema_duplicate_experiment.design = ?2 and experiment_id = ?3
            /*source*/
    '''
    )


GET_EX_XLM_ALL = (
    '''
    SELECT experiment_id, ema_parameter.name, parameter_value
            FROM ema_parameter JOIN ema_experiment_parameter on ema_experiment_parameter.parameter_id = ema_parameter.rowid
            JOIN ema_experiment ON ema_experiment_parameter.experiment_id = ema_experiment.rowid
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE s.name =?1
    UNION
    SELECT experiment_id, ema_measure.name, measure_value
            FROM ema_experiment_measure JOIN ema_measure on ema_experiment_measure.measure_id = ema_measure.rowid
            JOIN ema_experiment ON ema_experiment_measure.experiment_id = ema_experiment.rowid
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE s.name =?1
    '''
    )

GET_EX_XLM_ALL_BYSOURCE = GET_EX_XLM_ALL + ' AND ema_experiment_measure.measure_source =?2'

GET_EX_M_ALL = (
    '''
    SELECT experiment_id, ema_measure.name, measure_value
            FROM ema_experiment_measure JOIN ema_measure on ema_experiment_measure.measure_id = ema_measure.rowid
            JOIN ema_experiment ON ema_experiment_measure.experiment_id = ema_experiment.rowid
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE s.name =?1
    '''
    )

GET_EX_M_BY_ID_ALL = (
    '''
    SELECT experiment_id, ema_measure.name, measure_value
            FROM ema_experiment_measure JOIN ema_measure on ema_experiment_measure.measure_id = ema_measure.rowid
            JOIN ema_experiment ON ema_experiment_measure.experiment_id = ema_experiment.rowid
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE s.name =?1 AND experiment_id = ?2
    '''
    )

CREATE_META_MODEL = (
    '''
    INSERT INTO meta_model(scope_id, measure_id, lr_r2, gpr_cv, rmse)
        SELECT s.rowid, ema_measure.rowid, ?, ?, ? FROM
        ema_scope s JOIN ema_measure
        WHERE s.name = ? AND ema_measure.name = ?
    '''
    )

GET_META_MODEL = (
    '''
    SELECT lr_r2, gpr_cv, rmse
        FROM meta_model mm JOIN ema_scope s ON mm.scope_id = s.rowid
        JOIN ema_measure ON mm.measure_id = ema_measure.rowid
        WHERE s.name = ? AND ema_measure.name = ?
    '''
    )

UPDATE_META_MODEL = (
    '''
    UPDATE meta_model
        SET lr_r2 = ?, gpr_cv = ?, rmse = ?
    WHERE EXISTS
        (SELECT * FROM  meta_model mm
        JOIN ema_scope s ON mm.scope_id = s.rowid
        JOIN ema_measure ON mm.measure_id = ema_measure.rowid
        WHERE s.name = ? AND ema_measure.name = ?)
    '''
    )

ADD_MM_COEFF = (
    '''
    INSERT OR REPLACE INTO meta_model_param( scope_id, measure_id, parameter_id, est, std_error, pvalue )
        SELECT s.rowid, ema_measure.rowid, ema_parameter.rowid, ?, ?, ? FROM
            ema_scope s JOIN ema_measure JOIN ema_parameter
            WHERE s.name = ? AND ema_measure.name = ? AND ema_parameter.name = ?
    '''
    )

GET_MM_COEFF = (
    '''SELECT ema_parameter.name, est, std_error, pvalue
       FROM meta_model_param mmp JOIN meta_model mm
           ON (mmp.scope_id = mm.scope_id AND mmp.measure_id = mm.measure_id)
           JOIN ema_scope s ON mm.scope_id = s.rowid
           JOIN ema_measure ON mm.measure_id = ema_measure.rowid
           JOIN ema_parameter ON mmp.parameter_id = ema_parameter.rowid
        WHERE s.name = ? AND ema_measure.name = ?
    '''
    )

GET_SCOPE_NAMES = (
    '''SELECT name 
        FROM ema_scope 
        ORDER BY name;
    '''
)

GET_SCOPES_CONTAINING_DESIGN_NAME = (
    '''SELECT DISTINCT s.name
            FROM ema_experiment
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE ema_experiment.design =?
            ORDER BY s.name;
    '''
)


GET_DESIGN_NAMES = (
    '''SELECT DISTINCT ema_experiment.design
            FROM ema_experiment
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE s.name =?;
    '''
)

GET_EXPERIMENT_IDS_IN_DESIGN = (
    '''
    SELECT ema_experiment.rowid
        FROM ema_experiment
        JOIN ema_scope s ON ema_experiment.scope_id = s.rowid
        WHERE s.name =?1
        AND ema_experiment.design = ?2;
    '''
)

GET_EXPERIMENT_IDS_ALL = (
    '''
    SELECT ema_experiment.rowid
        FROM ema_experiment
        JOIN ema_scope s ON ema_experiment.scope_id = s.rowid
        WHERE s.name =?1;
    '''
)

INSERT_METAMODEL_PICKLE = (
    '''INSERT OR REPLACE INTO meta_model_pickles ( scope_id, metamodel_id, name, pickled_mm )
            SELECT ema_scope.rowid, ?2, ?3, ?4
            FROM ema_scope WHERE ema_scope.name = ?1
    '''
    )

GET_METAMODEL_PICKLE = (
    '''
	SELECT meta_model_pickles.name, meta_model_pickles.pickled_mm
		FROM meta_model_pickles
		JOIN ema_scope s ON meta_model_pickles.scope_id = s.rowid
		WHERE s.name =?1 AND meta_model_pickles.metamodel_id =?2;
	'''
)

GET_METAMODEL_IDS = (
    '''
	SELECT meta_model_pickles.metamodel_id
		FROM meta_model_pickles
		JOIN ema_scope s ON meta_model_pickles.scope_id = s.rowid
		WHERE s.name =?1 AND meta_model_pickles.pickled_mm NOT NULL ;
	'''
)

GET_NEW_METAMODEL_ID = (
    # '''
	# SELECT MAX(IFNULL(MAX(meta_model_pickles.metamodel_id), 0), IFNULL(MAX(meta_model_pickles.rowid), 0))+1
	# 	FROM meta_model_pickles;
	# '''
    '''
	SELECT IFNULL(MAX(meta_model_pickles.metamodel_id), 0)+1
		FROM meta_model_pickles;
	'''
)


GET_BOX_THRESHOLDS = (
    '''
    SELECT 
        ema_parameter.name, 
        threshold_value,
        threshold_type
    FROM
        ema_box_parameter
        JOIN ema_scope_box 
            ON ema_scope_box.box_id = ema_box_parameter.box_id
        JOIN ema_parameter
            ON ema_parameter.rowid = ema_box_parameter.parameter_id
        JOIN ema_scope_parameter
            ON ema_scope_parameter.parameter_id = ema_box_parameter.parameter_id
        JOIN ema_scope
            ON ema_scope.rowid = ema_scope_parameter.scope_id
        WHERE
            ema_scope.name = ?1
        AND
            ema_scope_box.box_name = ?2
    
    UNION ALL
    
    SELECT 
        ema_measure.name, 
        threshold_value,
        threshold_type
    FROM
        ema_box_measure
        JOIN ema_scope_box 
            ON ema_scope_box.box_id = ema_box_measure.box_id
        JOIN ema_measure
            ON ema_measure.rowid = ema_box_measure.measure_id
        JOIN ema_scope_measure
            ON ema_scope_measure.measure_id = ema_box_measure.measure_id
        JOIN ema_scope
            ON ema_scope.rowid = ema_scope_measure.scope_id
        WHERE
            ema_scope.name = ?1
        AND
            ema_scope_box.box_name = ?2
    '''
)

INSERT_BOX = (
    """
    INSERT OR REPLACE INTO ema_scope_box (parent_box_id, scope_id, box_name)
    SELECT null, ema_scope.rowid, ?2
    FROM ema_scope 
    WHERE ema_scope.name = ?1
    """
)

INSERT_SUBBOX = (
    """
    INSERT OR REPLACE INTO ema_scope_box (parent_box_id, scope_id, box_name)
    SELECT parent.box_id, ema_scope.rowid, ?2
    FROM ema_scope 
    JOIN ema_scope_box parent 
        ON parent.scope_id = ema_scope.rowid AND parent.box_name = ?3
    WHERE ema_scope.name = ?1
    """
)

GET_BOX_NAMES = (
    """
    SELECT DISTINCT
        ema_scope_box.box_name
    FROM
        ema_scope_box
        JOIN ema_scope
            ON ema_scope.rowid = ema_scope_box.scope_id
    WHERE
        ema_scope.name = ?1
    """
)

GET_BOX_PARENT_NAMES = (
    """
    SELECT
        child.box_name, parent.box_name
    FROM
        ema_scope_box child
        JOIN ema_scope
            ON ema_scope.rowid = child.scope_id
        JOIN ema_scope_box parent
            ON parent.box_id = child.parent_box_id
    WHERE
        ema_scope.name = ?1
    """
)

GET_BOX_PARENT_NAME = (
    """
    SELECT
        parent.box_name
    FROM
        ema_scope_box child
        JOIN ema_scope
            ON ema_scope.rowid = child.scope_id
        JOIN ema_scope_box parent
            ON parent.box_id = child.parent_box_id
    WHERE
        ema_scope.name = ?1
        AND child.box_name = ?2
    """
)

CLEAR_BOX_THRESHOLD_P = (
	'''
	DELETE FROM ema_box_parameter
	WHERE EXISTS (
		SELECT 
			*
		FROM
			ema_box_parameter
			JOIN ema_scope_box 
				ON ema_scope_box.box_id = ema_box_parameter.box_id
			JOIN ema_parameter
				ON ema_parameter.rowid = ema_box_parameter.parameter_id
			JOIN ema_scope_parameter
				ON ema_scope_parameter.parameter_id = ema_box_parameter.parameter_id
			JOIN ema_scope
				ON ema_scope.rowid = ema_scope_parameter.scope_id
			WHERE
				ema_scope.name = ?1
			AND
				ema_scope_box.box_name = ?2
			AND 
			    ema_parameter.name = ?3
	);
	'''
)

CLEAR_BOX_THRESHOLD_M = (
	'''
	DELETE FROM ema_box_measure
	WHERE EXISTS (
		SELECT 
			*
		FROM
			ema_box_measure
			JOIN ema_scope_box 
				ON ema_scope_box.box_id = ema_box_measure.box_id
			JOIN ema_measure
				ON ema_measure.rowid = ema_box_measure.measure_id
			JOIN ema_scope_measure
				ON ema_scope_measure.measure_id = ema_box_measure.measure_id
			JOIN ema_scope
				ON ema_scope.rowid = ema_scope_measure.scope_id
			WHERE
				ema_scope.name = ?1
			AND
				ema_scope_box.box_name = ?2
			AND 
			    ema_measure.name = ?3
	);
	'''
)

SET_BOX_THRESHOLD_P = (
    '''
    INSERT OR REPLACE INTO ema_box_parameter (
        box_id, 
        parameter_id, 
        threshold_value, 
        threshold_type
    )
    SELECT 
        ema_scope_box.box_id,
        ema_parameter.rowid,
        ?4,
        ?5
    FROM
        ema_scope_box 
        JOIN ema_parameter
		JOIN ema_scope
			ON ema_scope.rowid = ema_scope_box.scope_id
        
        WHERE ema_scope.name = ?1
        AND ema_scope_box.box_name = ?2
        AND ema_parameter.name = ?3
    '''
)

SET_BOX_THRESHOLD_M = (
    '''
    INSERT OR REPLACE INTO ema_box_measure (
        box_id, 
        measure_id, 
        threshold_value, 
        threshold_type
    )
    SELECT 
        ema_scope_box.box_id,
        ema_measure.rowid,
        ?4,
        ?5
    FROM
        ema_scope_box 
        JOIN ema_measure
		JOIN ema_scope
			ON ema_scope.rowid = ema_scope_box.scope_id
        WHERE ema_scope.name = ?1
        AND ema_scope_box.box_name = ?2
        AND ema_measure.name = ?3
    '''
)