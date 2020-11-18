# -*- coding: utf-8 -*-

import unittest
import pytest

import os
import emat
from emat.scope.scope import Scope, ScopeError
from emat.scope.box import Box, ChainedBox, Boxes
from emat import package_file
from emat.database.sqlite.sqlite_db import SQLiteDB
from emat import config


class TestScopeMethods(unittest.TestCase):
    ''' 
        tests parsing scope file    
    '''
    #  
    # one time test setup
    #
    scope_file = emat.package_file("model", "tests", "model_test.yaml")

    db_test = SQLiteDB(
        config.get("test_db_filename", ":memory:"),
        initialize=True,
    )

    #
    # Tests
    #
    def test_dump_scope(self):
        scp = Scope(self.scope_file)
        dumped = scp.dump()
        # print("="*40)
        # print(dumped)
        # print("="*40)
        loaded = Scope(scope_def=dumped, scope_file="fake/filename.yaml")
        assert loaded == scp # filename is intentionally different but let it go
        # but everything else is the same
        assert loaded.name == scp.name
        assert loaded.get_measures() == scp.get_measures()
        assert loaded.get_parameters() == scp.get_parameters()
        assert loaded.scope_file != scp.scope_file
        assert loaded.scope_file == "fake/filename.yaml"

        # fix name, still get equality
        loaded.scope_file = scp.scope_file
        assert loaded == scp

    def test_save_scope(self):
        scp = Scope(self.scope_file)
        scp.store_scope(self.db_test)
        
    def test_null_scope(self):
        scp = Scope(None)
        assert repr(scp) == "<emat.Scope with no content>"
        assert len(scp.get_measures()) == 0
        assert len(scp.get_parameters()) == 0

    def test_box(self):
        scope = Scope(package_file('model','tests','road_test.yaml'))

        with pytest.raises(TypeError):
            s = Box(scope=scope)

        s = Box(name="Speedy", scope=scope)
        s.set_upper_bound('build_travel_time', 70)
        with pytest.raises(ScopeError):
            s.set_upper_bound('not_a_thing', 70)
        assert len(s) == 1
        assert 'build_travel_time' in s
        assert s.parent_box_name is None

        s2 = Box(name="Notable", scope=scope, parent="Speedy")
        s2.set_lower_bound('expand_capacity', 20)
        assert len(s2) == 1
        assert 'build_travel_time' not in s2
        assert s2.parent_box_name == 'Speedy'

    def test_box_universe(self):
        scope = Scope(package_file('model','tests','road_test.yaml'))

        s = Box(name="Speedy", scope=scope)
        s.set_upper_bound('build_travel_time', 70)

        s2 = Box(name="Notable", scope=scope, parent="Speedy")
        s2.set_lower_bound('expand_capacity', 20)

        u = Boxes(s, s2, scope=scope)
        assert u.fancy_names() == ['Scope: EMAT Road Test', '▶ Speedy', '▷ ▶ Notable']
        assert u.plain_names() == [None, 'Speedy', 'Notable']

    def test_read_write_box(self):
        scope = Scope(package_file('model','tests','road_test.yaml'))
        db = SQLiteDB()
        scope.store_scope(db)

        s1 = Box(name="Speedy", scope=scope)
        s1.set_upper_bound('build_travel_time', 70)
        s1.relevant_features.add('debt_type')

        s2 = Box(name="Notable", scope=scope, parent="Speedy")
        s2.set_lower_bound('expand_capacity', 20)

        db.write_box(s1)
        db.write_box(s2)

        s1_ = db.read_box(scope.name, "Speedy")
        s2_ = db.read_box(scope.name, "Notable")

        assert s1 == s1_
        assert s2 == s2_
        assert s1.thresholds == s1_.thresholds
        assert s2.thresholds == s2_.thresholds
        assert s1.relevant_features == s1_.relevant_features
        assert s2.relevant_features == s2_.relevant_features

    def test_read_write_boxes(self):
        scope = Scope(package_file('model','tests','road_test.yaml'))
        db = SQLiteDB()
        scope.store_scope(db)

        s1 = Box(name="Speedy", scope=scope)
        s1.set_upper_bound('build_travel_time', 70)

        s2 = Box(name="Notable", scope=scope, parent="Speedy")
        s2.set_lower_bound('expand_capacity', 20)

        u = Boxes(s1, s2, scope=scope)

        db.write_boxes(u)

        scope2 = Scope(package_file('model','tests','road_test.yaml'))
        u2 = db.read_boxes(scope=scope2)

        assert u == u2
        assert u["Notable"].parent_box_name == u2["Notable"].parent_box_name

        s1_ = db.read_box(scope.name, "Speedy")
        s2_ = db.read_box(scope.name, "Notable")

        assert s1 == s1_
        assert s2 == s2_
        assert s1.relevant_features == s1_.relevant_features
        assert s2.relevant_features == s2_.relevant_features


if __name__ == '__main__':
    unittest.main()
   
