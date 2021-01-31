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

UPDATE_SCOPE_CONTENT = '''
    UPDATE 
        ema_scope
    SET 
        content = @scope_pickle
    WHERE 
        name = @scope_name
'''


GET_SCOPE = (
    '''SELECT content FROM ema_scope WHERE name = ?'''
)

DELETE_SCOPE = (
    ''' DELETE from ema_scope WHERE name = ?
    '''
    )

INSERT_SCOPE_XL = '''
    INSERT INTO 
        ema_scope_parameter( scope_id, parameter_id )
    SELECT 
        ema_scope.scope_id, 
        ema_parameter.parameter_id 
    FROM
        ema_scope 
        JOIN ema_parameter
    WHERE 
        ema_scope.name = ? 
        AND ema_parameter.name = ?
'''


INSERT_SCOPE_M = (
    '''INSERT INTO ema_scope_measure( scope_id, measure_id )
        SELECT ema_scope.scope_id, ema_measure.measure_id FROM
            ema_scope JOIN ema_measure
            WHERE ema_scope.name = ? AND ema_measure.name = ?
    '''
    )

GET_SCOPE_XL = (
    '''SELECT ema_parameter.name
        FROM ema_parameter JOIN ema_scope_parameter sv ON (ema_parameter.parameter_id = sv.parameter_id)
        JOIN ema_scope s ON (sv.scope_id = s.scope_id)
        WHERE s.name = ?
    '''
    )

GET_SCOPE_X = '''
    SELECT 
        ema_parameter.name
    FROM 
        ema_parameter 
        JOIN ema_scope_parameter sv 
            ON (ema_parameter.parameter_id = sv.parameter_id)
        JOIN ema_scope s 
            ON (sv.scope_id = s.scope_id)
        WHERE 
            s.name = ?
            AND ema_parameter.ptype = 1
'''


GET_SCOPE_L = (
    '''SELECT ema_parameter.name
        FROM ema_parameter JOIN ema_scope_parameter sv ON (ema_parameter.parameter_id = sv.parameter_id)
        JOIN ema_scope s ON (sv.scope_id = s.scope_id)
        WHERE s.name = ?
        AND ema_parameter.ptype = 0
    '''
    )

GET_SCOPE_C = (
    '''SELECT ema_parameter.name
        FROM ema_parameter JOIN ema_scope_parameter sv ON (ema_parameter.parameter_id = sv.parameter_id)
        JOIN ema_scope s ON (sv.scope_id = s.scope_id)
        WHERE s.name = ?
        AND ema_parameter.ptype = 2
    '''
    )

GET_SCOPE_M = (
    '''SELECT ema_measure.name
        FROM ema_measure JOIN ema_scope_measure sp ON (ema_measure.measure_id = sp.measure_id)
        JOIN ema_scope s ON (sp.scope_id = s.scope_id)
        WHERE s.name = ?
    '''
    )

INSERT_EX = (
    '''INSERT INTO ema_experiment ( scope_id, design )
            SELECT ema_scope.scope_id, ?
            FROM ema_scope WHERE ema_scope.name = ?
    '''
    )

INSERT_DESIGN = '''
INSERT OR IGNORE INTO ema_design (scope_id, design) 
    SELECT ema_scope.scope_id, ?2
    FROM ema_scope WHERE ema_scope.name = ?1
'''

INSERT_EXPERIMENT = '''
    INSERT INTO ema_experiment ( scope_id )
        SELECT ema_scope.scope_id
        FROM ema_scope WHERE ema_scope.name = ?
'''

INSERT_EXPERIMENT_WITH_ID = '''
    INSERT INTO ema_experiment ( experiment_id, scope_id )
        SELECT ?2, ema_scope.scope_id
        FROM ema_scope WHERE ema_scope.name = ?1
'''


INSERT_DESIGN_EXPERIMENT = '''
    INSERT OR IGNORE INTO ema_design_experiment (experiment_id, design_id)
        SELECT ?3, d.design_id
        FROM ema_design d
        JOIN ema_scope s ON (d.scope_id = s.scope_id)
        WHERE d.design = ?2
        AND s.name = ?1

'''

NEW_EXPERIMENT_RUN = '''
    INSERT INTO 
        ema_experiment_run ( 
            run_id, 
            experiment_id, 
            run_status, 
            run_valid, 
            run_location, 
            run_source ) 
    VALUES ( 
        @run_id, 
        @experiment_id, 
        'init', 
        1, 
        @run_location, 
        @run_source )
'''



DELETE_DESIGN_EXPERIMENTS = '''
    DELETE FROM ema_design_experiment
    WHERE ema_design_experiment.design_id IN (
        SELECT ema_design.design_id
        FROM ema_design 
        JOIN ema_scope s ON (ema_design.scope_id = s.scope_id)
        WHERE s.name = ? AND ema_design.design = ?
    )
'''

DELETE_LOOSE_EXPERIMENTS = '''
    DELETE FROM ema_experiment
    WHERE ema_experiment.experiment_id NOT IN (
        SELECT edd.experiment_id
        FROM ema_design_experiment edd
        JOIN ema_design ed ON (ed.design_id = edd.design_id)
        JOIN ema_scope s ON (ed.scope_id = s.scope_id)
        WHERE s.name = ?
    )
'''

DELETE_MEASURES_BY_EXPERIMENT_ID = '''
    DELETE FROM main.ema_experiment_measure
    WHERE ema_experiment_measure.experiment_id IN (?)
'''

DELETE_RUN_ID = '''
    DELETE FROM ema_experiment_run
    WHERE ema_experiment_run.run_id = @run_id
'''

INVALIDATE_RUN_ID = '''
    UPDATE 
        ema_experiment_run
    SET
        run_valid = 0
    WHERE 
        ema_experiment_run.run_id = @run_id
        AND run_valid != 0
'''



INSERT_EX_XL = (
    '''INSERT INTO ema_experiment_parameter( experiment_id, parameter_id, parameter_value )
        SELECT ?, ema_parameter.parameter_id, ? FROM
            ema_parameter WHERE ema_parameter.name = ?
    '''
    )

GET_EXPERIMENT_PARAMETERS = '''
    SELECT 
        eep.experiment_id, ep.name, parameter_value
    FROM 
        ema_experiment_parameter eep
        JOIN ema_parameter ep 
            ON eep.parameter_id = ep.parameter_id     -- convert parameter_id to name
        JOIN ema_experiment ee 
            ON eep.experiment_id = ee.experiment_id   -- connect to experiment table to allow filtering
        JOIN ema_scope s 
            ON ee.scope_id = s.scope_id               -- connect to scope table, filter on matching scope
        JOIN ema_design_experiment ede 
            ON ee.experiment_id = ede.experiment_id   -- make design_id available
        JOIN ema_design ed 
            ON (s.scope_id = ed.scope_id AND ede.design_id = ed.design_id)              
        WHERE 
            s.name = @scope_name 
            AND ed.design = @design_name
'''

GET_EXPERIMENT_IDS_BY_VALUE = '''
    SELECT 
        eep.experiment_id
    FROM 
        ema_experiment_parameter eep
        JOIN ema_parameter ep
            ON eep.parameter_id = ep.parameter_id
        JOIN ema_experiment ee
            ON eep.experiment_id = ee.experiment_id
        JOIN ema_scope s 
            ON ee.scope_id = s.scope_id
    WHERE 
        s.name =?1 
        AND ep.name = ?2 
        AND parameter_value = ?3;
'''


GET_EXPERIMENT_IDS_BY_DESIGN_AND_VALUE = '''
    SELECT 
        eep.experiment_id
    FROM 
        ema_experiment_parameter eep
        JOIN ema_parameter ep
            ON eep.parameter_id = ep.parameter_id
        JOIN ema_experiment ee
            ON eep.experiment_id = ee.experiment_id
        JOIN ema_scope s 
            ON ee.scope_id = s.scope_id
    WHERE 
        s.name =?1 
        AND ee.design = ?2 
        AND ep.name = ?3 
        AND parameter_value = ?4;
'''



GET_EX_XL_ALL = '''
    SELECT 
        eep.experiment_id, 
        ep.name, 
        parameter_value
    FROM 
        ema_experiment_parameter eep
        JOIN ema_parameter ep
            ON eep.parameter_id = ep.parameter_id
        JOIN ema_experiment ee
            ON eep.experiment_id = ee.experiment_id
        JOIN ema_scope s 
            ON ee.scope_id = s.scope_id
    WHERE 
        s.name = @scope_name;
'''


GET_EX_XL_IDS_IN = '''
    SELECT 
        eep.experiment_id, 
        ep.name, 
        parameter_value
    FROM ema_experiment_parameter eep
        JOIN ema_parameter ep
            ON eep.parameter_id = ep.parameter_id
        JOIN ema_experiment ee
            ON eep.experiment_id = ee.experiment_id
        JOIN ema_scope s 
            ON ee.scope_id = s.scope_id
    WHERE 
        s.name =?1 
        AND eep.experiment_id in (???);
'''







INSERT_EX_M = '''
    REPLACE INTO ema_experiment_measure ( 
        experiment_id, 
        measure_id, 
        measure_value, 
        measure_run )
    SELECT 
        @experiment_id, 
        ema_measure.measure_id, 
        @measure_value, 
        eer.run_rowid
    FROM 
        ema_measure 
        JOIN ema_experiment_run eer
            ON eer.run_id = @measure_run
    WHERE ema_measure.name = @measure_name
'''

_DEBUG_INSERT_EX_M = '''
    SELECT 
        @experiment_id, 
        ema_measure.measure_id, 
        @measure_value, 
        eer.run_rowid
    FROM 
        ema_measure 
        LEFT JOIN ema_experiment_run eer
            ON eer.run_id = @measure_run
    WHERE ema_measure.name = @measure_name
'''


GET_EXPERIMENT_PARAMETERS_AND_MEASURES = '''
    SELECT eep.experiment_id, ep.name, parameter_value
        FROM ema_parameter ep
        JOIN ema_experiment_parameter eep on eep.parameter_id = ep.parameter_id
        JOIN ema_experiment ee ON eep.experiment_id = ee.experiment_id
        JOIN ema_scope s on ee.scope_id = s.scope_id
        JOIN ema_design_experiment ede ON ee.experiment_id = ede.experiment_id
        JOIN ema_design ed ON (s.scope_id = ed.scope_id AND ed.design_id = ede.design_id)
        WHERE s.name =?1 and ed.design = ?2
    UNION
    SELECT eem.experiment_id, ema_measure.name, measure_value
        FROM ema_experiment_measure eem JOIN ema_measure on eem.measure_id = ema_measure.measure_id
        JOIN ema_experiment ee ON eem.experiment_id = ee.experiment_id
        JOIN ema_scope es on ee.scope_id = es.scope_id
        JOIN ema_design_experiment ede ON ee.experiment_id = ede.experiment_id
        JOIN ema_design ed ON (es.scope_id = ed.scope_id AND ed.design_id = ede.design_id)
        WHERE es.name =?1 and ed.design = ?2
        /*source*/
'''

GET_EXPERIMENT_PARAMETERS_AND_MEASURES_BYSOURCE = GET_EXPERIMENT_PARAMETERS_AND_MEASURES.replace(
    '/*source*/',
    ' AND eem.measure_source =?3'
)






GET_EXPERIMENT_MEASURES_MASTER = '''
    SELECT DISTINCT 
        eem.experiment_id, --index_type
        runs.run_id,
        ema_measure.name, 
        measure_value,
        runs.run_source,
        runs.run_rowid,
        runs.experiment_id as run_ex_id
    FROM 
        ema_experiment_measure eem 
        JOIN ema_measure 
            ON eem.measure_id = ema_measure.measure_id
        JOIN ema_experiment ee 
            ON eem.experiment_id = ee.experiment_id
        JOIN ema_scope es 
            ON ee.scope_id = es.scope_id
        JOIN ema_design_experiment ede 
            ON ee.experiment_id = ede.experiment_id
        JOIN ema_design ed 
            ON (es.scope_id = ed.scope_id AND ed.design_id = ede.design_id)
        JOIN /* most recent valid run with results matching target source */ (
            SELECT
                *,
                max(run_timestamp)
            FROM
                ema_experiment_run
            WHERE
                (
                    run_rowid IN (
                        SELECT DISTINCT measure_run 
                        FROM ema_experiment_measure eem3 
                        WHERE eem3.measure_value IS NOT NULL
                    )
                )
                AND run_valid = 1
                AND run_source = @measure_source
            GROUP BY
                experiment_id, run_source
        ) /* end most recent */ runs 
            ON runs.run_rowid = eem.measure_run
        WHERE 
            es.name = @scope_name  
            AND ed.design = @design_name 
            AND eem.experiment_id = @experiment_id
            AND measure_value IS NOT NULL
            AND run_source = @measure_source
            AND run_valid = 1
'''



GET_EX_XLM_ALL = (
    '''
    SELECT 
        eep.experiment_id, ep.name, parameter_value
    FROM 
        ema_parameter ep
        JOIN ema_experiment_parameter eep
            ON eep.parameter_id = ep.parameter_id
        JOIN ema_experiment ee
            ON eep.experiment_id = ee.experiment_id
        JOIN ema_scope s 
            ON ee.scope_id = s.scope_id
        WHERE 
            s.name =?1
    UNION
    SELECT 
        eem.experiment_id, em.name, measure_value
    FROM 
        ema_experiment_measure eem
        JOIN ema_measure em 
            ON eem.measure_id = em.measure_id
        JOIN ema_experiment ee 
            ON eem.experiment_id = ee.experiment_id
        JOIN ema_scope s 
            ON ee.scope_id = s.scope_id
        WHERE 
            s.name =?1
    '''
    )

GET_EX_XLM_ALL_BYSOURCE = GET_EX_XLM_ALL + ' AND ema_experiment_measure.measure_source =?2'

GET_EXPERIMENT_MEASURE_SOURCES = '''
    SELECT DISTINCT
        eer.run_source
    FROM 
        ema_experiment_measure eem
        JOIN ema_measure em
            ON eem.measure_id = em.measure_id
        JOIN ema_experiment ee
            ON eem.experiment_id = ee.experiment_id
        JOIN ema_scope es 
            ON ee.scope_id = es.scope_id
        JOIN ema_experiment_run eer
            ON eem.measure_run = eer.run_rowid
        /*by-design-join*/
        WHERE 
            es.name = @scope_name
            AND measure_value IS NOT NULL
            /*by-design-where*/
'''

GET_EXPERIMENT_MEASURE_SOURCES_BY_DESIGN = GET_EXPERIMENT_MEASURE_SOURCES.replace("/*by-design-join*/", '''            
        JOIN ema_design_experiment ede 
            ON ee.experiment_id = ede.experiment_id
        JOIN ema_design ed 
            ON (es.scope_id = ed.scope_id AND ed.design_id = ede.design_id)
''').replace("/*by-design-where*/", '''   
        AND ed.design = @design_name         
''')


CREATE_META_MODEL = (
    '''
    INSERT INTO meta_model(scope_id, measure_id, lr_r2, gpr_cv, rmse)
        SELECT s.scope_id, ema_measure.measure_id, ?, ?, ? FROM
        ema_scope s JOIN ema_measure
        WHERE s.name = ? AND ema_measure.name = ?
    '''
    )

GET_META_MODEL = (
    '''
    SELECT lr_r2, gpr_cv, rmse
        FROM meta_model mm JOIN ema_scope s ON mm.scope_id = s.scope_id
        JOIN ema_measure ON mm.measure_id = ema_measure.measure_id
        WHERE s.name = ? AND ema_measure.name = ?
    '''
    )

UPDATE_META_MODEL = (
    '''
    UPDATE meta_model
        SET lr_r2 = ?, gpr_cv = ?, rmse = ?
    WHERE EXISTS
        (SELECT * FROM  meta_model mm
        JOIN ema_scope s ON mm.scope_id = s.scope_id
        JOIN ema_measure ON mm.measure_id = ema_measure.measure_id
        WHERE s.name = ? AND ema_measure.name = ?)
    '''
    )

ADD_MM_COEFF = (
    '''
    INSERT OR REPLACE INTO meta_model_param( scope_id, measure_id, parameter_id, est, std_error, pvalue )
        SELECT s.scope_id, ema_measure.measure_id, ema_parameter.parameter_id, ?, ?, ? FROM
            ema_scope s JOIN ema_measure JOIN ema_parameter
            WHERE s.name = ? AND ema_measure.name = ? AND ema_parameter.name = ?
    '''
    )

GET_MM_COEFF = (
    '''SELECT ema_parameter.name, est, std_error, pvalue
       FROM meta_model_param mmp JOIN meta_model mm
           ON (mmp.scope_id = mm.scope_id AND mmp.measure_id = mm.measure_id)
           JOIN ema_scope s ON mm.scope_id = s.scope_id
           JOIN ema_measure ON mm.measure_id = ema_measure.measure_id
           JOIN ema_parameter ON mmp.parameter_id = ema_parameter.parameter_id
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
            JOIN ema_scope s on ema_design.scope_id = s.scope_id
            WHERE ema_design.design =?
            ORDER BY s.name;
    '''
)


GET_DESIGN_NAMES = '''
    SELECT DISTINCT ema_design.design
        FROM ema_design
        JOIN ema_scope s on ema_design.scope_id = s.scope_id
        WHERE s.name =?;
'''


GET_EXPERIMENT_IDS_IN_DESIGN = (
    '''
    SELECT ema_experiment.experiment_id
        FROM ema_experiment
        JOIN ema_scope s ON ema_experiment.scope_id = s.scope_id
        JOIN ema_design_experiment de ON ema_experiment.experiment_id = de.experiment_id
        JOIN ema_design d ON de.design_id = d.design_id
        WHERE s.name =?1
        AND d.design = ?2;
    '''
)

GET_EXPERIMENT_IDS_ALL = (
    '''
    SELECT ema_experiment.experiment_id
        FROM ema_experiment
        JOIN ema_scope s ON ema_experiment.scope_id = s.scope_id
        WHERE s.name =?1;
    '''
)

INSERT_METAMODEL_PICKLE = (
    '''INSERT OR REPLACE INTO meta_model_pickles ( scope_id, metamodel_id, name, pickled_mm )
            SELECT ema_scope.scope_id, ?2, ?3, ?4
            FROM ema_scope WHERE ema_scope.name = ?1
    '''
    )

GET_METAMODEL_PICKLE = (
    '''
	SELECT meta_model_pickles.name, meta_model_pickles.pickled_mm
		FROM meta_model_pickles
		JOIN ema_scope s ON meta_model_pickles.scope_id = s.scope_id
		WHERE s.name =?1 AND meta_model_pickles.metamodel_id =?2;
	'''
)

GET_METAMODEL_IDS = (
    '''
	SELECT meta_model_pickles.metamodel_id
		FROM meta_model_pickles
		JOIN ema_scope s ON meta_model_pickles.scope_id = s.scope_id
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
            ON ema_parameter.parameter_id = ema_box_parameter.parameter_id
        JOIN ema_scope_parameter
            ON ema_scope_parameter.parameter_id = ema_box_parameter.parameter_id
        JOIN ema_scope
            ON ema_scope.scope_id = ema_scope_parameter.scope_id
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
            ON ema_measure.measure_id = ema_box_measure.measure_id
        JOIN ema_scope_measure
            ON ema_scope_measure.measure_id = ema_box_measure.measure_id
        JOIN ema_scope
            ON ema_scope.scope_id = ema_scope_measure.scope_id
        WHERE
            ema_scope.name = ?1
        AND
            ema_scope_box.box_name = ?2
    '''
)

INSERT_BOX = (
    """
    INSERT OR REPLACE INTO ema_scope_box (parent_box_id, scope_id, box_name)
    SELECT null, ema_scope.scope_id, ?2
    FROM ema_scope 
    WHERE ema_scope.name = ?1
    """
)

INSERT_SUBBOX = (
    """
    INSERT OR REPLACE INTO ema_scope_box (parent_box_id, scope_id, box_name)
    SELECT parent.box_id, ema_scope.scope_id, ?2
    FROM ema_scope 
    JOIN ema_scope_box parent 
        ON parent.scope_id = ema_scope.scope_id AND parent.box_name = ?3
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
            ON ema_scope.scope_id = ema_scope_box.scope_id
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
            ON ema_scope.scope_id = child.scope_id
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
            ON ema_scope.scope_id = child.scope_id
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
				ON ema_parameter.parameter_id = ema_box_parameter.parameter_id
			JOIN ema_scope_parameter
				ON ema_scope_parameter.parameter_id = ema_box_parameter.parameter_id
			JOIN ema_scope
				ON ema_scope.scope_id = ema_scope_parameter.scope_id
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
				ON ema_measure.measure_id = ema_box_measure.measure_id
			JOIN ema_scope_measure
				ON ema_scope_measure.measure_id = ema_box_measure.measure_id
			JOIN ema_scope
				ON ema_scope.scope_id = ema_scope_measure.scope_id
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
        ema_parameter.parameter_id,
        ?4,
        ?5
    FROM
        ema_scope_box 
        JOIN ema_parameter
		JOIN ema_scope
			ON ema_scope.scope_id = ema_scope_box.scope_id
        
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
        ema_measure.measure_id,
        ?4,
        ?5
    FROM
        ema_scope_box 
        JOIN ema_measure
		JOIN ema_scope
			ON ema_scope.scope_id = ema_scope_box.scope_id
        WHERE ema_scope.name = ?1
        AND ema_scope_box.box_name = ?2
        AND ema_measure.name = ?3
    '''
)


UPDATE_DATABASE_ema_design_experiment = (
    "PRAGMA foreign_keys = OFF",

    '''
        INSERT OR IGNORE INTO ema_design ( scope_id, design )
        SELECT DISTINCT scope_id, design FROM ema_experiment;
    ''',

    '''          
        INSERT OR IGNORE INTO ema_design_experiment ( experiment_id, design_id )
        SELECT ema_experiment.experiment_id, ema_design.design_id
        FROM ema_experiment
        JOIN ema_design ON ema_design.design = ema_experiment.design;
    ''',

)

UPDATE_DATABASE_ema_experiment_measure_ADD_measure_run = (
    '''
        ALTER TABLE ema_experiment_measure
        ADD COLUMN measure_run UUID;
    ''',
)

UPDATE_DATABASE_ema_experiment_run_ADD_run_source = (
    '''
        ALTER TABLE ema_experiment_run
        ADD COLUMN run_source INT NOT NULL DEFAULT 0;
    ''',
)


from ... import __version__
import numpy as np
__version_as_int__ = np.asarray([
    int(i)
    for i in __version__.replace("a",'').replace("b",'').split(".")
]) @ np.asarray([1000000,1000,1])
SET_VERSION_DATABASE = f'''
INSERT OR IGNORE INTO ema_tool_info VALUES ('version', {__version_as_int__});
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
