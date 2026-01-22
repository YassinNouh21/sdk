# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from feast import FeatureStore


class FeastClient:
    """Client for Feast feature store operations.

    Feast is a feature store that enables offline retrieval of historical datasets
    and online serving of features/data for ML applications.

    Requires the feast package to be installed. Install it with:

        pip install 'kubeflow[feast]'

    """

    def __init__(self, repo_path: str | None = None, config: dict[str, Any] | None = None):
        """Initialize the FeastClient.

        Args:
            repo_path: Path to the Feast repository. If not provided, uses the current directory.
            config: Optional configuration dictionary for Feast FeatureStore.
                   If provided, takes precedence over repo_path.

        Raises:
            ImportError: If feast is not installed.
        """
        try:
            from feast import FeatureStore
        except ImportError as e:
            raise ImportError(
                "feast is not installed. Install it with:\n\n"  # fmt: skip
                "  pip install 'kubeflow[feast]'\n"
            ) from e

        if config is not None:
            self._store: FeatureStore = FeatureStore(config=config)
        else:
            self._store: FeatureStore = FeatureStore(repo_path=repo_path)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: list[str],
        full_feature_names: bool = False,
    ) -> pd.DataFrame:
        """Retrieve historical features for training datasets.

        Args:
            entity_df: DataFrame with entity keys and timestamps.
            features: List of feature references in the format "feature_view:feature_name".
            full_feature_names: Whether to use full feature names in the output DataFrame.

        Returns:
            DataFrame with historical feature values joined to the entity_df.
        """
        return self._store.get_historical_features(
            entity_df=entity_df,
            features=features,
            full_feature_names=full_feature_names,
        ).to_df()

    def get_online_features(
        self,
        features: list[str],
        entity_rows: list[dict[str, Any]],
        full_feature_names: bool = False,
    ) -> dict[str, list[Any]]:
        """Retrieve online features for real-time inference.

        Args:
            features: List of feature references in the format "feature_view:feature_name".
            entity_rows: List of entity dictionaries with entity keys.
            full_feature_names: Whether to use full feature names in the output.

        Returns:
            Dictionary mapping feature names to lists of feature values.
        """
        result = self._store.get_online_features(
            features=features,
            entity_rows=entity_rows,
            full_feature_names=full_feature_names,
        )
        return result.to_dict()

    def apply(self, objects: list[Any] | None = None) -> None:
        """Apply changes to the feature store.

        This method deploys feature definitions to the feature store,
        including feature views, entities, and data sources.

        Args:
            objects: List of Feast objects (Feature Views, Entities, Data Sources) to apply.
                    If None or empty list, applies all objects defined in the repository.
        """
        if objects is None:
            objects = []
        self._store.apply(objects)

    def materialize(
        self,
        start_date: Any,
        end_date: Any,
        feature_views: list[str] | None = None,
    ) -> None:
        """Materialize features into the online store.

        Args:
            start_date: Start date for materialization (datetime or string).
            end_date: End date for materialization (datetime or string).
            feature_views: Optional list of feature view names to materialize.
                          If None, all feature views are materialized.
        """
        self._store.materialize(
            start_date=start_date,
            end_date=end_date,
            feature_views=feature_views,
        )

    def materialize_incremental(
        self, end_date: Any, feature_views: list[str] | None = None
    ) -> None:
        """Materialize features incrementally into the online store.

        This method materializes features from the last materialized state up to end_date.

        Args:
            end_date: End date for materialization (datetime or string).
            feature_views: Optional list of feature view names to materialize.
                          If None, all feature views are materialized.
        """
        self._store.materialize_incremental(
            end_date=end_date,
            feature_views=feature_views,
        )

    def list_feature_views(self) -> list[Any]:
        """List all feature views in the feature store.

        Returns:
            List of feature view objects.
        """
        return self._store.list_feature_views()

    def list_entities(self) -> list[Any]:
        """List all entities in the feature store.

        Returns:
            List of entity objects.
        """
        return self._store.list_entities()

    def list_data_sources(self) -> list[Any]:
        """List all data sources in the feature store.

        Returns:
            List of data source objects.
        """
        return self._store.list_data_sources()

    @property
    def store(self) -> FeatureStore:
        """Access the underlying Feast FeatureStore instance.

        Returns:
            The Feast FeatureStore instance.
        """
        return self._store
