import pandas as pd
import numpy as np
import pmdarima as pm
from pmdarima.arima import ndiffs, nsdiffs
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from statsmodels.tsa.api import ExponentialSmoothing # Import ETS model
import warnings

warnings.filterwarnings("ignore")

# --- Global Configuration ---
CONFIG = {
    'file_path': 'dummy_timeseries.csv',
    'output_csv_path': 'model_selection_results.csv',
    'target_column': 'value',
    'exog_columns': ['month_name', 'weekday_name', 'holiday'],
    'forecast_horizons': [7, 14, 30],
    'n_splits': 3,
    'min_train_size': 50,
    'seasonal_period': 7,
    'p_range': [0, 2],
    'q_range': [0, 2],
    'P_range': [0, 2],
    'Q_range': [0, 2],
    'max_lag': 14,
    'xgboost_params': {
        'objective': 'reg:squarederror',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 10
    }
}

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100

class TimeSeriesModelSelector:
    def __init__(self, config):
        self.config = config
        self.target_column = self.config['target_column']
        self.exog_columns = self.config['exog_columns']
        self.file_path = self.config['file_path']
        self.df = None
        self.y = None
        self.X = None
        self.d = None
        self.D = None
        self.results = []
        self._load_and_prepare_data()
        self._determine_differencing_orders()

    def _load_and_prepare_data(self):
        print("Step 1: Loading and preparing data...")
        df_raw = pd.read_csv(self.file_path)
        df_raw.columns = df_raw.columns.str.strip()
        self.df = df_raw.set_index(pd.to_datetime(df_raw['date'])).asfreq('D')
        self.df[self.target_column] = self.df[self.target_column].fillna(method='ffill')
        self.df = self.df.dropna(subset=[self.target_column])
        self.y = self.df[self.target_column]
        self.X = pd.get_dummies(self.df[self.exog_columns], drop_first=True, dtype=int)
        print("Data loaded successfully.\n")

    def _determine_differencing_orders(self):
        print("Step 2: Determining optimal differencing orders...")
        self.d = ndiffs(self.y, test='adf')
        self.D = nsdiffs(self.y, m=self.config['seasonal_period'], test='ocsb')
        print(f"Determined orders: d={self.d}, D={self.D} for m={self.config['seasonal_period']}\n")

    def _create_ml_features(self, data, max_lag):
        features = {}
        for lag in range(1, max_lag + 1):
            features[f'lag_{lag}'] = data.shift(lag)
        features['rolling_mean_7'] = data.rolling(window=7).mean()
        features['rolling_std_7'] = data.rolling(window=7).std()
        return pd.DataFrame(features)

    def run_evaluation(self):
        print("Step 3: Starting model evaluation...")
        self._evaluate_naive_models()
        self._evaluate_ets() ## NEW ##
        self._evaluate_sarimax()
        self._evaluate_xgboost()
        self._display_results()

    def _evaluate_model(self, model_name, fit_predict_func):
        ## UPDATED to return results for each horizon ##
        horizons = self.config['forecast_horizons']
        n_splits = self.config['n_splits']
        horizon_results = []
        
        print(f"\n--- Evaluating {model_name} ---")
        for horizon in horizons:
            metrics = {'rmse': [], 'mae': [], 'smape': []}
            print(f"  Testing horizon: {horizon} days")
            for i in range(n_splits):
                split_point = len(self.y) - horizon * (n_splits - i)
                if split_point < self.config['min_train_size']: continue
                
                y_train, y_test = self.y.iloc[:split_point], self.y.iloc[split_point:split_point + horizon]
                X_train, X_test = self.X.iloc[:split_point], self.X.iloc[split_point:split_point + horizon]
                
                try:
                    predictions = fit_predict_func(y_train, y_test, X_train, X_test)
                    if len(predictions) != len(y_test): continue
                    metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, predictions)))
                    metrics['mae'].append(mean_absolute_error(y_test, predictions))
                    metrics['smape'].append(symmetric_mean_absolute_percentage_error(y_test, predictions))
                except Exception as e:
                    print(f"    Error during evaluation on split {i+1}: {e}")
            
            if metrics['rmse']:
                horizon_results.append({
                    'Model': model_name,
                    'Horizon': horizon,
                    'Avg RMSE': np.mean(metrics['rmse']),
                    'Avg MAE': np.mean(metrics['mae']),
                    'Avg sMAPE': np.mean(metrics['smape']),
                    'Parameters': '-'
                })
        return horizon_results

    def _evaluate_naive_models(self):
        ## UPDATED to add results from all naive models for per-horizon comparison ##
        def fit_predict_seasonal(y_train, y_test, X_train, X_test):
            m = self.config['seasonal_period']
            last_season = y_train.iloc[-m:]
            reps = int(np.ceil(len(y_test) / m))
            return np.tile(last_season, reps)[:len(y_test)]
        
        def fit_predict_last(y_train, y_test, X_train, X_test):
            return np.repeat(y_train.iloc[-1], len(y_test))

        def fit_predict_median(y_train, y_test, X_train, X_test):
            return np.repeat(y_train.median(), len(y_test))
            
        models = {
            "Seasonal Naive": fit_predict_seasonal, "Last Value Naive": fit_predict_last, "Median Naive": fit_predict_median
        }
        for name, func in models.items():
            results = self._evaluate_model(name, func)
            if results: self.results.extend(results)

    def _evaluate_ets(self):
        ## NEW ## Function to evaluate the ETS model
        def fit_predict_ets(y_train, y_test, X_train, X_test):
            # ETS doesn't use exogenous variables, so we only need y_train
            model = ExponentialSmoothing(
                y_train,
                seasonal_periods=self.config['seasonal_period'],
                trend='add',
                seasonal='add'
            ).fit()
            return model.forecast(len(y_test))
        
        results = self._evaluate_model("ETS", fit_predict_ets)
        if results: self.results.extend(results)

    def _evaluate_sarimax(self):
        best_params_per_horizon = {}
        def fit_predict_sarimax(y_train, y_test, X_train, X_test):
            horizon = len(y_test)
            arima_model = pm.auto_arima(
                y=y_train, X=X_train, d=self.d, D=self.D, m=self.config['seasonal_period'],
                start_p=self.config['p_range'][0], max_p=self.config['p_range'][1],
                start_q=self.config['q_range'][0], max_q=self.config['q_range'][1],
                start_P=self.config['P_range'][0], max_P=self.config['P_range'][1],
                start_Q=self.config['Q_range'][0], max_Q=self.config['Q_range'][1],
                trace=False, error_action='ignore', suppress_warnings=True, stepwise=True
            )
            best_params_per_horizon[horizon] = f"order={arima_model.order}, seasonal_order={arima_model.seasonal_order}"
            return arima_model.predict(n_periods=len(y_test), X=X_test)

        results = self._evaluate_model("SARIMAX", fit_predict_sarimax)
        if results:
            for res in results:
                res['Parameters'] = best_params_per_horizon.get(res['Horizon'], '-')
            self.results.extend(results)
    
    def _evaluate_xgboost(self):
        def fit_predict_xgboost(y_train, y_test, X_train, X_test):
            max_lag = self.config['max_lag']
            ml_features_train = self._create_ml_features(y_train, max_lag)
            X_train_full = pd.concat([X_train, ml_features_train], axis=1)
            combined_train = pd.concat([y_train, X_train_full], axis=1).dropna()
            y_train_final = combined_train[self.target_column]
            X_train_final = combined_train.drop(self.target_column, axis=1)

            validation_size = len(y_test)
            X_train_fit, X_val = X_train_final[:-validation_size], X_train_final[-validation_size:]
            y_train_fit, y_val = y_train_final[:-validation_size], y_train_final[-validation_size:]
            
            model = xgb.XGBRegressor(**self.config['xgboost_params'])
            model.fit(X_train_fit, y_train_fit, eval_set=[(X_val, y_val)], verbose=False)

            history_y = y_train.copy()
            predictions = []
            for i in range(len(y_test)):
                ml_features_step = self._create_ml_features(history_y, max_lag)
                X_step_exog = X_test.iloc[[i]]
                X_step_ml = ml_features_step.iloc[[-1]]
                X_step_full = pd.concat([X_step_exog.reset_index(drop=True), X_step_ml.reset_index(drop=True)], axis=1)
                pred = model.predict(X_step_full)[0]
                predictions.append(pred)
                history_y = pd.concat([history_y, pd.Series([pred], index=[y_test.index[i]])])
            return np.array(predictions)
            
        results = self._evaluate_model("XGBoost", fit_predict_xgboost)
        if results: self.results.extend(results)

    def _display_results(self):
        ## UPDATED to display results grouped by horizon ##
        print("\n" + "="*50)
        print("--- Final Model Comparison (Per Horizon) ---")
        print("="*50)
        if not self.results:
            print("No models were successfully evaluated.")
            return
            
        results_df = pd.DataFrame(self.results)
        
        for horizon in sorted(results_df['Horizon'].unique()):
            print(f"\n--- Forecast Horizon: {horizon} days ---")
            horizon_df = results_df[results_df['Horizon'] == horizon].sort_values(by=['Avg RMSE', 'Avg MAE', 'Avg sMAPE'])
            print(horizon_df.to_string(index=False))
            
            best_model = horizon_df.iloc[0]
            print(f"\n🏆 Best Model for {horizon}-day Horizon: {best_model['Model']}")
            print(f"   - RMSE: {best_model['Avg RMSE']:.4f}")
            if best_model['Model'] == 'SARIMAX':
                print(f"   - Best Parameters Found: {best_model['Parameters']}")

        try:
            results_df.to_csv(self.config['output_csv_path'], index=False)
            print(f"\n\n✅ Full results successfully saved to '{self.config['output_csv_path']}'")
        except Exception as e:
            print(f"\n\n❌ Failed to save results to CSV: {e}")

if __name__ == '__main__':
    model_selector = TimeSeriesModelSelector(config=CONFIG)
    model_selector.run_evaluation()