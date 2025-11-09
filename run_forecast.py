from src.main_forecast import run_all
if __name__ == '__main__':
    run_all('data/example_kaggle_rainfall.csv', forecast_steps=48)
