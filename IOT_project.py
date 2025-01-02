import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Charger le modèle et les objets de prétraitement
with open("model_IOT.pkl", "rb") as model_file:
    model = pickle.load(model_file)
print("Nombre de caractéristiques attendues par le modèle :", model.n_features_in_)

with open("preprocessing.pkl", "rb") as preprocess_file:
    preprocessing_objects = pickle.load(preprocess_file)
    
labelencoder = preprocessing_objects['labelencoder']
labelencoder_y = preprocessing_objects['labelencoder_y']
scaler = preprocessing_objects['scaler']
top_features = preprocessing_objects['top_features']  # Charger les 30 caractéristiques importantes

# Interface Streamlit
st.image("ehtp.png", use_column_width=True)
st.title("IoT Attack Type Prediction")
st.sidebar.image("iot.jpg", use_column_width=True)

st.markdown("""
### Instructions :
1. Chargez un fichier CSV de données de validation.
2. Cliquez sur "Prédire" pour voir les types d'attaques prédits (et leur ID "No"), les probabilités, et si la prédiction est correcte.
""")

# Chargement du fichier de validation
uploaded_file = st.file_uploader("Choisissez un fichier CSV de validation", type="csv")

if uploaded_file is not None:
    # Charger les données de validation
    validation_data = pd.read_csv(uploaded_file)
    
    # Vérifier si 'Attack_type' existe dans le fichier chargé
    if 'Attack_type' in validation_data.columns:
        y_true = validation_data['Attack_type']
        validation_data = validation_data.drop(columns=['Attack_type'])
    else:
        y_true = None  # Pas de comparaison possible

    # Vérifier si la colonne "no" existe
    if 'no' in validation_data.columns:
        # Extraire la colonne "no"
        no_column = validation_data['no']
    else:
        st.error("La colonne 'no' est manquante dans le fichier CSV.")
        no_column = None  # Assurez-vous que 'no' existe, sinon nous allons avoir un problème en la manipulant plus tard.
    
    # Gérer les valeurs inconnues pour 'proto' et 'service'
    for col in ['proto', 'service']:
        validation_data[col] = validation_data[col].apply(
            lambda x: x if x in labelencoder.classes_ else "unknown"
        )
    
    # Ajouter une classe "unknown" si nécessaire
    if "unknown" not in labelencoder.classes_:
        labelencoder.classes_ = np.append(labelencoder.classes_, "unknown")

    # Encoder 'proto' et 'service'
    validation_data['proto'] = labelencoder.transform(validation_data['proto'])
    validation_data['service'] = labelencoder.transform(validation_data['service'])

    # Sélectionner uniquement les 30 caractéristiques importantes
    validation_data = validation_data[top_features]
    
    # Appliquer la normalisation sur les caractéristiques
    validation_data = scaler.transform(validation_data)
    
    # Faire des prédictions
    predictions = model.predict(validation_data)
    probabilities = model.predict_proba(validation_data)
    predicted_classes = labelencoder_y.inverse_transform(predictions)
    
    # Créer un DataFrame pour afficher les résultats
    results = pd.DataFrame({
        "Predicted Class": predicted_classes,
        "Prediction Probability": [round(max(prob), 4) for prob in probabilities],
    })

    # Ajouter la colonne "no" si elle existe
    if no_column is not None:
        results["no"] = no_column.values  # Assurez-vous d'utiliser .values ou .to_numpy() pour éviter des erreurs

    # Ajouter la colonne 'True Class' et 'Correct Prediction'
    if y_true is not None:
        y_true_encoded = labelencoder_y.transform(y_true)
        true_classes = labelencoder_y.inverse_transform(y_true_encoded)
        results["True Class"] = true_classes
        results["Correct Prediction"] = results["Predicted Class"] == results["True Class"]
    else:
        # Si 'Attack_type' n'est pas présent, ajouter "no" dans la colonne "True Class"
        results["True Class"] = "no"
        results["Correct Prediction"] = False  # Impossible de vérifier la prédiction sans la vérité de terrain

    # Afficher les résultats
    st.write("### Résultats des prédictions")
    st.write(results)

    # Résumé des performances
    if y_true is not None:
        accuracy = (results["Correct Prediction"].sum() / len(results)) * 100
        st.write(f"### Précision globale : {accuracy:.2f}%")
else:
    st.info("Veuillez charger un fichier CSV de validation.")
