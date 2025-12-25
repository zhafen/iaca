import pandas as pd
import pandera.pandas as pa
from sklearn.base import BaseEstimator, TransformerMixin
from .. import data


# Custom transformer for adding error and validity metadata columns
class LogPrepper(BaseEstimator, TransformerMixin):
    """
    Transformer that adds two columns to a DataFrame:
    - 'errors': for tracking errors on a per-row basis (default: empty string)
    - 'valid': for indicating if a row is valid (default: True)
    """

    def fit(self, _X, _y=None):
        # Stateless transformer, nothing to fit
        return self

    def transform(
        self, X: pd.DataFrame, _registry: data.Registry = None
    ) -> pd.DataFrame:
        X = X.copy()
        if "errors" not in X.columns:
            X["errors"] = ""
        if "is_valid" not in X.columns:
            X["is_valid"] = True
        return X


class ComponentNormalizer(BaseEstimator, TransformerMixin):

    def fit(self, _X, _y=None):
        return self

    def transform(self, X, registry: data.Registry = None):
        """
        Metadata
        --------
        - todo:
            value: >
                The way this is set up currently, no components can be a value of dict
                type. No issue for now, but we may want to revisit this in the future.
            priority: 0.3
        """

        # This transform operates on unindexed DataFrames.
        # They'll be reindexed when they're returned to the registry.
        X = registry.reset_index(X)

        # Identify rows by type
        is_dict = X["component"].apply(lambda v: isinstance(v, dict))

        # If the "component" column is a non-dict values (both null and not) then that
        # is the value of the component, so we just rename it
        X_values = X.loc[~is_dict].rename(columns={"component": "value"})

        # We expand dictionary entries to get the different fields as columns
        X_dict = X.loc[is_dict].copy()
        # Only expand one level deep, so that nested dictionaries are preserved
        X_expanded = pd.json_normalize(X_dict["component"], max_level=1)
        X_expanded = X_expanded.set_index(X_dict.index)
        # Connect back to the other columns--entity and comp_key
        X_dict = X_dict.join(X_expanded).drop(columns=["component"])

        # Rejoin
        X_out = pd.concat([X_dict, X_values])

        return X_out


# Transformer to extract and validate component definitions (bottom of file)
class ComponentDefExtractor(BaseEstimator, TransformerMixin):

    def fit(self, _X, _y=None, registry: data.Registry = None):
        self.registry = registry
        return self

    def transform(self, X, registry: data.Registry = None):

        X = X.copy()

        # Add in all components defined in the registry
        # and mark the ones that are not defined
        X["is_defined"] = True
        registry_comps = pd.DataFrame({"entity": registry.keys()})
        X = X.merge(registry_comps, how="outer", on="entity")
        X.loc[X["is_defined"].isna(), "is_defined"] = False
        X["is_defined"] = X["is_defined"].astype(bool)

        # Parse the fields
        X = X.apply(self._parse_fields, axis="columns")

        # All component definitions include an "entity" column and a "comp_key" column
        # We pull the definitions from a "default_fields" component.
        default_fields = X.loc[X["entity"] == "default_fields", "fields"].iloc[0]
        X["fields"] = X["fields"].apply(lambda d: {**default_fields, **d})

        # Update validity with whether or not the component is defined
        X.loc[~X["is_defined"], "is_valid"] = False
        X.loc[~X["is_defined"], "errors"] += "Component definition does not exist. "

        # Reorganize and rename
        X = X.rename(
            columns={
                "fields.component": "unparsed_fields",
                "component.multiplicity": "multiplicity",
                "component.normalize": "normalize",
            }
        ).drop(columns=["component.value"])

        return X

    def _parse_fields(
        self,
        row,
    ) -> pd.Series:
        # If not given any fields then this is just a flag component
        if pd.isna(row["fields.component"]):
            row["fields.component"] = {}
            row["fields"] = {}
            row["is_valid"] = True
            row["errors"] = ""
            return row

        fields_i = {}
        valid_fields = True
        valid_message = ""
        for field_key, field_value in row["fields.component"].items():
            try:
                field = data.Field.from_kv_pair(field_key, field_value)
                fields_i[field.name] = field
            except (ValueError, TypeError) as e:
                valid_fields = False
                valid_message += (
                    f"Field {field_key} is incorrectly formatted: {field_value}. "
                    f"Error: {e}"
                )

        row["fields"] = fields_i
        row["is_valid"] = valid_fields
        row["errors"] = valid_message

        return row


class ComponentValidator(BaseEstimator, TransformerMixin):
    """

    Metadata
    ----------
    - todo: Add validation that warns users when they try to modify a system component.
    """

    def fit(self, _X, _y=None):
        return self

    def transform(self, X, registry: data.Registry = None):

        # This is another transform operation that operates on unindexed DataFrames.
        X = registry.reset_index(X)

        # Since X is the input view, which is what we're validating,
        # we can use the view_components attribute to get the name of the component
        key = X.attrs["view_components"]

        # Get the component definition
        # If we're validating compdef we have to handle it specially,
        # because X is not yet in the format it should be.
        if key == "compdef":
            component_def = X.loc[X["entity"] == "compdef"]
            if len(component_def) > 1:
                raise ValueError(
                    f"Multiple component definitions found for key '{key}'. "
                    "This should not happen, please check your registry."
                )
            component_def = component_def.iloc[0]
        else:
            component_def = registry["compdef"].loc[key]

        # Set the attributes for validity and errors
        X.attrs["is_valid"] = component_def["is_valid"]
        X.attrs["errors"] = component_def["errors"]

        # Validate X against the schema
        try:
            # The fields in the component definition are the schema for the DataFrame
            dataframe_schema = pa.DataFrameSchema(component_def["fields"])
            X = dataframe_schema.validate(X)
        except (AttributeError, pa.errors.SchemaError) as e:
            X.attrs["is_valid"] = False
            X.attrs["errors"] += str(e) + " "

        return X
