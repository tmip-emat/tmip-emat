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

INSERT_DESIGN = '''
INSERT OR IGNORE INTO ema_design (scope_id, design) 
    SELECT ema_scope.rowid, ?2
    FROM ema_scope WHERE ema_scope.name = ?1
'''

INSERT_EXPERIMENT = '''
    INSERT INTO ema_experiment ( scope_id )
        SELECT ema_scope.rowid
        FROM ema_scope WHERE ema_scope.name = ?
'''

INSERT_DESIGN_EXPERIMENT = '''
    INSERT OR IGNORE INTO ema_design_experiment (experiment_id, design_id)
        SELECT ?3, d.rowid
        FROM ema_design d
        JOIN ema_scope s ON (d.scope_id = s.rowid)
        WHERE d.design = ?2
        AND s.name = ?1

'''



DELETE_DESIGN_EXPERIMENTS = '''
    DELETE FROM ema_design_experiment
    WHERE ema_design_experiment.design_id IN (
        SELECT ema_design.rowid
        FROM ema_design 
        JOIN ema_scope s ON (ema_design.scope_id = s.rowid)
        WHERE s.name = ? AND ema_design.design = ?
    )
'''

DELETE_LOOSE_EXPERIMENTS = '''
    DELETE FROM ema_experiment
    WHERE ema_experiment.rowid NOT IN (
        SELECT edd.experiment_id
        FROM ema_design_experiment edd
        JOIN ema_design ed ON (ed.rowid = edd.design_id)
        JOIN ema_scope s ON (ed.scope_id = s.rowid)
        WHERE s.name = ?
    )
'''

DELETE_MEASURES_BY_EXPERIMENT_ID = '''
    DELETE FROM main.ema_experiment_measure
    WHERE ema_experiment_measure.experiment_id IN (?)
'''



INSERT_EX_XL = (
    '''INSERT INTO ema_experiment_parameter( experiment_id, parameter_id, parameter_value )
        SELECT ?, ema_parameter.rowid, ? FROM
            ema_parameter WHERE ema_parameter.name = ?
    '''
    )

GET_EXPERIMENT_PARAMETERS = '''
    SELECT 
        eep.experiment_id, ep.name, parameter_value
    FROM 
        ema_experiment_parameter eep
        JOIN ema_parameter ep ON eep.parameter_id = ep.rowid     -- convert parameter_id to name
        JOIN ema_experiment ee ON eep.experiment_id = ee.rowid   -- connect to experiment table to allow filtering
        JOIN ema_scope s on ee.scope_id = s.rowid                -- connect to scope table, filter on matching scope
        JOIN ema_design_experiment ede ON ee.rowid = ede.experiment_id  -- make design_id available
        JOIN ema_design ed ON (s.rowid = ed.scope_id AND ede.design_id = ed.rowid)              
        WHERE s.name =?1 and ed.design = ?2
'''

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


GET_PENDING_EXPERIMENT_PARAMETERS = '''
    SELECT eep.experiment_id, ep.name, eep.parameter_value
        FROM ema_experiment_parameter eep
        JOIN ema_parameter ep ON eep.parameter_id = ep.rowid
        JOIN ema_experiment ee ON eep.experiment_id = ee.rowid
        JOIN ema_scope es ON ee.scope_id = es.rowid
        JOIN ema_design_experiment ede ON ee.rowid = ede.experiment_id
        JOIN ema_design ed ON (es.rowid = ed.scope_id AND ed.rowid = ede.design_id)
        WHERE es.name =?1 AND ed.design = ?2
        AND eep.experiment_id NOT IN (
            SELECT eem2.experiment_id
            FROM ema_experiment_measure eem2
            JOIN ema_experiment ee2 ON eem2.experiment_id = ee2.rowid
            JOIN ema_scope es2 on ee2.scope_id = es2.rowid
            JOIN ema_design_experiment ede2 ON ee2.rowid = ede2.experiment_id
            JOIN ema_design ed2 ON (es2.rowid = ed2.scope_id AND ed2.rowid = ede2.design_id)
            WHERE es2.name =?1 AND ed2.design = ?2 AND eem2.measure_value IS NOT NULL
        )
'''


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

INSERT_EX_M = '''
    INSERT OR REPLACE INTO ema_experiment_measure ( experiment_id, measure_id, measure_value, measure_source )
    SELECT 
        ?1, ema_measure.rowid, ?2, ?3 
    FROM 
        ema_measure 
    WHERE ema_measure.name = ?4
'''


GET_EXPERIMENT_PARAMETERS_AND_MEASURES = '''
    SELECT eep.experiment_id, ep.name, parameter_value
        FROM ema_parameter ep
        JOIN ema_experiment_parameter eep on eep.parameter_id = ep.rowid
        JOIN ema_experiment ee ON eep.experiment_id = ee.rowid
        JOIN ema_scope s on ee.scope_id = s.rowid
        JOIN ema_design_experiment ede ON ee.rowid = ede.experiment_id
        JOIN ema_design ed ON (s.rowid = ed.scope_id AND ed.rowid = ede.design_id)
        WHERE s.name =?1 and ed.design = ?2
    UNION
    SELECT eem.experiment_id, ema_measure.name, measure_value
        FROM ema_experiment_measure eem JOIN ema_measure on eem.measure_id = ema_measure.rowid
        JOIN ema_experiment ee ON eem.experiment_id = ee.rowid
        JOIN ema_scope es on ee.scope_id = es.rowid
        JOIN ema_design_experiment ede ON ee.rowid = ede.experiment_id
        JOIN ema_design ed ON (es.rowid = ed.scope_id AND ed.rowid = ede.design_id)
        WHERE es.name =?1 and ed.design = ?2
        /*source*/
'''

GET_EXPERIMENT_PARAMETERS_AND_MEASURES_BYSOURCE = GET_EXPERIMENT_PARAMETERS_AND_MEASURES.replace(
    '/*source*/',
    ' AND eem.measure_source =?3'
)


GET_EXPERIMENT_MEASURES = '''
    SELECT 
        ede.experiment_id, 
        ema_measure.name, 
        measure_value
    FROM 
        ema_experiment_measure eem 
        JOIN ema_measure 
            ON eem.measure_id = ema_measure.rowid
        JOIN ema_experiment ee 
            ON eem.experiment_id = ee.rowid
        JOIN ema_scope es 
            ON ee.scope_id = es.rowid
        JOIN ema_design_experiment ede 
            ON ee.rowid = ede.experiment_id
        JOIN ema_design ed 
            ON (es.rowid = ed.scope_id AND ed.rowid = ede.design_id)
        WHERE 
            es.name =?1 
            AND ed.design = ?2 
            AND measure_value IS NOT NULL
            /*source*/
'''

GET_EXPERIMENT_MEASURES_BYSOURCE = GET_EXPERIMENT_MEASURES.replace('/*source*/', ' AND eem.measure_source =?3')



GET_EXPERIMENT_MEASURES_BY_ID = '''
    SELECT ede.experiment_id, ema_measure.name, measure_value
            FROM ema_experiment_measure eem JOIN ema_measure on eem.measure_id = ema_measure.rowid
            JOIN ema_experiment ee ON eem.experiment_id = ee.rowid
            JOIN ema_scope es on ee.scope_id = es.rowid
            JOIN ema_design_experiment ede ON ee.rowid = ede.experiment_id
            JOIN ema_design ed ON (es.rowid = ed.scope_id AND ed.rowid = ede.design_id)
            WHERE es.name =?1 and ed.design = ?2 AND eem.experiment_id = ?3 AND measure_value IS NOT NULL
            /*source*/
'''





GET_EX_XLM_ALL = (
    '''
    SELECT experiment_id, ema_parameter.name, parameter_value
            FROM ema_parameter JOIN ema_experiment_parameter on ema_experiment_parameter.parameter_id = ema_parameter.rowid
            JOIN ema_experiment ON ema_experiment_parameter.experiment_id = ema_experiment.rowid
            JOIN ema_scope s on ema_experiment.scope_id = s.rowid
            WHERE s.name =?1
    UNION
    SELECT 
        experiment_id, em.name, measure_value
    FROM 
        ema_experiment_measure eem
        JOIN ema_measure em 
            ON eem.measure_id = em.rowid
        JOIN ema_experiment ee 
            ON eem.experiment_id = ee.rowid
        JOIN ema_scope s 
            ON ee.scope_id = s.rowid
        WHERE 
            s.name =?1
    '''
    )

GET_EX_XLM_ALL_BYSOURCE = GET_EX_XLM_ALL + ' AND ema_experiment_measure.measure_source =?2'

GET_EXPERIMENT_MEASURE_SOURCES = '''
    SELECT DISTINCT
        measure_source
    FROM 
        ema_experiment_measure eem
        JOIN ema_measure em
            ON eem.measure_id = em.rowid
        JOIN ema_experiment ee
            ON eem.experiment_id = ee.rowid
        JOIN ema_scope es 
            ON ee.scope_id = es.rowid
        /*by-design-join*/
        WHERE 
            es.name = @scope_name
            AND measure_value IS NOT NULL
            /*by-design-where*/
'''

GET_EXPERIMENT_MEASURE_SOURCES_BY_DESIGN = GET_EXPERIMENT_MEASURE_SOURCES.replace("/*by-design-join*/", '''            
        JOIN ema_design_experiment ede 
            ON ee.rowid = ede.experiment_id
        JOIN ema_design ed 
            ON (es.rowid = ed.scope_id AND ed.rowid = ede.design_id)
''').replace("/*by-design-where*/", '''   
        AND ed.design = @design_name         
''')

GET_EX_M_ALL = '''
    SELECT DISTINCT
        experiment_id, 
        em.name, 
        measure_value
    FROM 
        ema_experiment_measure eem
        JOIN ema_measure em
            ON eem.measure_id = em.rowid
        JOIN ema_experiment ee
            ON eem.experiment_id = ee.rowid
        JOIN ema_scope s 
            ON ee.scope_id = s.rowid
        WHERE 
            s.name = ?1 
            AND measure_value IS NOT NULL
'''



GET_EX_M_BY_ID_ALL = '''
    SELECT DISTINCT
        experiment_id, 
        em.name, 
        measure_value
    FROM 
        ema_experiment_measure eem
        JOIN ema_measure em
            ON eem.measure_id = em.rowid
        JOIN ema_experiment ee
            ON eem.experiment_id = ee.rowid
        JOIN ema_scope s 
            ON ee.scope_id = s.rowid
    WHERE 
        s.name =?1 
        AND experiment_id = ?2 
        AND measure_value IS NOT NULL
'''

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
            FROM ema_design
            JOIN ema_scope s on ema_design.scope_id = s.rowid
            WHERE ema_design.design =?
            ORDER BY s.name;
    '''
)


GET_DESIGN_NAMES = '''
    SELECT DISTINCT ema_design.design
        FROM ema_design
        JOIN ema_scope s on ema_design.scope_id = s.rowid
        WHERE s.name =?;
'''


GET_EXPERIMENT_IDS_IN_DESIGN = (
    '''
    SELECT ema_experiment.rowid
        FROM ema_experiment
        JOIN ema_scope s ON ema_experiment.scope_id = s.rowid
        JOIN ema_design_experiment de ON ema_experiment.rowid = de.experiment_id
        JOIN ema_design d ON de.design_id = d.rowid
        WHERE s.name =?1
        AND d.design = ?2;
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


UPDATE_DATABASE = (
    '''
        INSERT OR IGNORE INTO ema_design ( scope_id, design )
        SELECT DISTINCT scope_id, design FROM ema_experiment;
    ''',

    '''          
        INSERT OR IGNORE INTO ema_design_experiment ( experiment_id, design_id )
        SELECT ema_experiment.rowid, ema_design.rowid
        FROM ema_experiment
        JOIN ema_design ON ema_design.design = ema_experiment.design;
    ''',
)

from ... import __version__
import numpy as np
SET_VERSION_DATABASE = f'''
INSERT OR IGNORE INTO ema_tool_info VALUES ('version', {np.asarray([int(i) for i in __version__.split(".")]) @ np.asarray([1000000,1000,1])});
'''
SET_MINIMUM_VERSION_DATABASE = f'''
INSERT OR IGNORE INTO ema_tool_info VALUES ('minimum_version', 4000); -- 0.4.0
'''


GET_VERSION_DATABASE = f'''
SELECT val FROM ema_tool_info WHERE tag='version'
'''

GET_MINIMUM_VERSION_DATABASE = f'''
SELECT val FROM ema_tool_info WHERE tag='minimum_version'
'''
