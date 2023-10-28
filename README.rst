you need to set a OPENAI_API_KEY environment variable with a valid key.

.. code-block:: bash

  source .flaskenv
  python3 -m venv env
  source env/bin/activate
  pip install -r requirements.txt 
  python app.py

should load up on http://127.0.0.1:5001
