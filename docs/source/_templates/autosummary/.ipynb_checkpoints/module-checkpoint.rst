{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:

{% block attributes %}
{% if attributes %}
.. rubric:: Module attributes

.. autosummary::
   :toctree:

{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
.. rubric:: Functions

.. autosummary::
   :toctree:
   :nosignatures:

{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
.. rubric:: Classes

.. autosummary::
   :toctree:
   :template: class.rst
   :nosignatures:

{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block exceptions %}
{% if exceptions %}
.. rubric:: Exceptions

.. autosummary::
   :toctree:

{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Submodules

.. autosummary::
   :toctree:
   :template: module.rst
   :recursive:

{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
