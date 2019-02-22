
.. py:currentmodule:: emat.model

GBNRTC Model
------------

.. autoclass:: GBNRTCModel
    :show-inheritance:
    :members:
    :exclude-members:
        setup, run, load_measures, post_process, get_experiment_archive_path, archive,
        start_transcad, run_model, model_init



Implementation-Specific Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These methods are defined specifically in this GBNRTC-specific class,
in order to interface with the TransCAD model for the GBNRTC region.

.. automethod:: GBNRTCModel.setup
.. automethod:: GBNRTCModel.start_transcad
.. automethod:: GBNRTCModel.run
.. automethod:: GBNRTCModel.run_model
.. automethod:: GBNRTCModel.load_measures
.. automethod:: GBNRTCModel.post_process
.. automethod:: GBNRTCModel.get_experiment_archive_path
.. automethod:: GBNRTCModel.archive
