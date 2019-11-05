
from .. import package_file, Scope, PythonCoreModel, SQLiteDB

def road_test(*args, yamlfile='road_test.yaml', **kwargs):
	road_test_scope_file = package_file('model', 'tests', yamlfile)
	s = Scope(road_test_scope_file)
	db = SQLiteDB(*args, **kwargs)
	if s.name not in db.read_scope_names():
		s.store_scope(db)
	from ..model.core_python import Road_Capacity_Investment
	m = PythonCoreModel(Road_Capacity_Investment, scope=s, db=db)
	return s, db, m

def gbnrtc(*args, **kwargs):
	scope_file = package_file('examples', 'gbnrtc', 'gbnrtc_scope.yaml')
	s = Scope(scope_file)
	db = SQLiteDB(*args, **kwargs)
	s.store_scope(db)
	from ..model.core_files.gbnrtc_model import GBNRTCModel
	cfg_file = package_file('examples', 'gbnrtc', 'gbnrtc_model_config.yaml')
	m = GBNRTCModel(configuration=cfg_file, scope=s, db=db)
	return s, db, m

