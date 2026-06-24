from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Cargar el pipeline guardado
pipeline = joblib.load('pipeline_mejor.pkl')

# Columnas que espera el modelo
COLUMNAS = ['sqft_living', 'grade', 'condition', 'long', 'zipcode']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener valores del formulario
        sqft_living  = float(request.form['sqft_living'])
        grade        = int(request.form['grade'])
        condition    = int(request.form['condition'])
        long_val     = float(request.form['long'])
        zipcode      = int(request.form['zipcode'])

        # Validaciones básicas
        errores = []
        if sqft_living <= 0:
            errores.append("La superficie habitable debe ser mayor a 0.")
        if not (1 <= grade <= 13):
            errores.append("El grado de calidad debe estar entre 1 y 13.")
        if not (1 <= condition <= 5):
            errores.append("La condición debe estar entre 1 y 5.")
        if not (-122.6 <= long_val <= -121.0):
            errores.append("La longitud debe estar en el rango de King County (-122.6 a -121.0).")
        if zipcode < 98001 or zipcode > 98199:
            errores.append("El código postal debe ser un ZIP válido de King County (98001-98199).")

        if errores:
            return render_template('index.html',
                                   error=True,
                                   mensajes_error=errores,
                                   sqft_living=sqft_living,
                                   grade=grade,
                                   condition=condition,
                                   long_val=long_val,
                                   zipcode=zipcode)

        # Crear DataFrame con los mismos nombres de columna que el modelo espera
        entrada = pd.DataFrame([[sqft_living, grade, condition, long_val, zipcode]],
                                columns=COLUMNAS)

        # Predecir
        prediccion = pipeline.predict(entrada)[0]
        precio_formateado = f"${prediccion:,.2f}"

        return render_template('index.html',
                               prediccion=precio_formateado,
                               sqft_living=sqft_living,
                               grade=grade,
                               condition=condition,
                               long_val=long_val,
                               zipcode=zipcode)

    except ValueError:
        return render_template('index.html',
                               error=True,
                               mensajes_error=["Por favor ingresa solo valores numéricos en todos los campos."])
    except Exception as e:
        return render_template('index.html',
                               error=True,
                               mensajes_error=[f"Error inesperado: {str(e)}"])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
