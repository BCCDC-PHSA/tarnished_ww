from .io import standardize_input, validate_population
from .schemas import ColumnSpec
from .build_functions import build_joint_model

def fit_joint_model(df,
                pop_df,
                diseases = ["covid","rsv","flua"],
                draws = 1000,
                tunes = 1000,
                target_accept = 0.9,
                chains = 4,
                cols = ColumnSpec()):
    # Preprocess data
    df_train = standardize_input(df, diseases, cols)
    population = validate_population(pop_df, df_train[cols.region_internal].unique())




build_joint_model(diseases, df_train, y_ed, pop.population.values, pop.shape[0], tests_per_capita = total_tests_per_capita)
