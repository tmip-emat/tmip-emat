
.. py:currentmodule:: emat.database

Databases
=========

.. toctree::
    :hidden:

    abstract
    sqlitedb
    database-walkthrough

The :class:`Database` provides a abstract interface structure
for interacting with databases used in storing the inputs and outputs of
exploratory analysis. The specific database system that backs up these
commands is not explicitly defined here.

.. rubric:: :doc:`SQLite Database <sqlitedb>`

To demonstrate a possible implementation, the TMIP-EMAT package
includes a :class:`SQLiteDB` class, which allows for the creation
of a database for storing and retrieving results using SQLite,
which is a free, lightweight, server-less database that requires
no configuration.

A :doc:`walkthrough <database-walkthrough>` guide is available, showing how
to import and export data from a database instance, and how
to handle re-running the core model and storing revised results
when incorrect data has accidentally been saved.

