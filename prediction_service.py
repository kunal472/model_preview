import pandas as pd
import joblib
import os

class PredictionService:
    def __init__(self, model_dir='models'):
        """
        Loads all necessary model artifacts from the specified directory.
        """
        print("Loading prediction service artifacts...")
        self.crop_recommender = joblib.load(os.path.join(model_dir, 'crop_recommender.pkl'))
        self.crop_scaler = joblib.load(os.path.join(model_dir, 'crop_recommender_scaler.pkl'))
        
        self.yield_forecaster = joblib.load(os.path.join(model_dir, 'yield_forecaster.pkl'))
        self.yield_scaler = joblib.load(os.path.join(model_dir, 'yield_forecaster_scaler.pkl'))
        self.yield_model_columns = joblib.load(os.path.join(model_dir, 'yield_forecaster_columns.pkl'))

        self.sustainability_scores = pd.read_csv(os.path.join(model_dir, 'sustainability_scores.csv')).set_index('label')
        self.cost_of_cultivation = joblib.load(os.path.join(model_dir, 'cost_of_cultivation.pkl'))
        print("✅ Artifacts loaded successfully.")

    def _get_top_recommendations(self, live_features, n=5):
        """
        Gets the top N crop recommendations based on environmental factors.
        """
        # Prepare features for the recommender model
        recommender_features = pd.DataFrame({
            'N': [live_features.get('topsoil_nitrogen')],
            'P': [live_features.get('P_placeholder', 50)], # Add placeholder if not from API
            'K': [live_features.get('K_placeholder', 50)], # Add placeholder if not from API
            'temperature': [live_features.get('avg_temp_celsius')],
            'humidity': [live_features.get('avg_humidity_percent')],
            'ph': [live_features.get('topsoil_phh2o')],
            'rainfall': [live_features.get('total_rainfall_mm')]
        })
        
        # Scale the features
        recommender_features_scaled = self.crop_scaler.transform(recommender_features)
        
        # Get probabilities for each crop
        probabilities = self.crop_recommender.predict_proba(recommender_features_scaled)
        
        # Get the top N recommendations
        top_n_indices = probabilities.argsort()[0][-n:][::-1]
        top_n_crops = self.crop_recommender.classes_[top_n_indices]
        
        return top_n_crops

    def get_final_recommendations(self, live_features):
        """
        Generates final ranked recommendations with yield, profit, and sustainability.
        """
        top_crops = self._get_top_recommendations(live_features)
        
        final_results = []
        
        # This price is a placeholder. A real system would fetch prices for each recommended crop.
        market_price_per_quintal = live_features.get('avg_modal_price', 2500) # Default price if not found

        for crop in top_crops:
            # 1. Prepare features for the yield model
            yield_features = live_features.copy()
            
            # Create a dataframe with all required columns, initialized to zero
            encoded_df = pd.DataFrame(columns=self.yield_model_columns)
            encoded_df.loc[0] = 0
            
            # Set the one-hot encoded column for the current crop to 1
            crop_column_name = f'label_{crop}'
            if crop_column_name in encoded_df.columns:
                encoded_df[crop_column_name] = 1

            # Populate the environmental features
            # Note: Ensure the keys in live_features match the column names in your training data
            for col in encoded_df.columns:
                 if col in live_features:
                     encoded_df.loc[0, col] = live_features[col]
            
            # 2. Scale and Predict Yield
            yield_features_scaled = self.yield_scaler.transform(encoded_df)
            predicted_yield = self.yield_forecaster.predict(yield_features_scaled)[0]
            
            # 3. Look up other metrics
            sustainability = self.sustainability_scores.loc[crop, 'sustainability_score']
            cost = self.cost_of_cultivation.get(crop, 0) # Get cost, default to 0 if not found
            
            # 4. Calculate Profit
            gross_revenue = predicted_yield * market_price_per_quintal
            profit_margin = gross_revenue - cost
            
            final_results.append({
                'crop': crop.capitalize(),
                'predicted_yield_quintal_per_hectare': round(predicted_yield, 2),
                'sustainability_score_10': round(sustainability, 2),
                'estimated_profit_rs_per_hectare': round(profit_margin, 2)
            })
            
        return final_results