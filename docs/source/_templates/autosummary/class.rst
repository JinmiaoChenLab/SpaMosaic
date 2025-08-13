{{ fullname | escape | underline}}

.. autoclass:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

{% block methods %}
{% if methods %}
.. rubric:: Methods

.. autosummary::
   :toctree:
   :nosignatures:

{% for item in methods %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
.. rubric:: Attributes

.. autosummary::
   :toctree:
   :nosignatures:

{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
