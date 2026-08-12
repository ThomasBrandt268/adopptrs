# API reference

Generated from the docstrings. This is an appendix, not an entry point:
the modules are documented individually, but the reasoning behind them
lives in [Pipeline](pipeline.md) and [Training](training.md).

:::{note}
Convention: **docstrings are public documentation and are written in
English**; inline comments are working notes and may be in any language.
The modules exposed below follow it. Command-line scripts under `misc/`
and `tests/` still carry French docstrings, which are their usage notes
rather than API surface.
:::

Building this page requires the training environment to be importable
(`torch`, `opencv-python`, `pyproj`), since `autodoc` imports each module.
If Sphinx runs outside that environment, the sections below will be empty
and the rest of the documentation remains unaffected.

## Data

```{eval-rst}
.. automodule:: dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: via
   :members:
   :undoc-members:
```

## Models and criteria

```{eval-rst}
.. automodule:: models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: criterions
   :members:
   :undoc-members:
```

## Imagery services

```{eval-rst}
.. automodule:: wms
   :members:
   :undoc-members:

.. automodule:: walonmap
   :members:
   :undoc-members:
```

## Post-processing

```{eval-rst}
.. automodule:: summarize
   :members:
   :undoc-members:
```
