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

"""Tests for FeastClient."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from unittest.mock import MagicMock

import pytest

from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


@pytest.fixture(autouse=True)
def skip_if_no_feast():
    """Skip tests if feast not installed."""
    pytest.importorskip("feast")


@pytest.fixture
def mock_feast_store():
    """Create a mock FeatureStore with all methods we wrap."""
    store = MagicMock()
    # Set up return values for list methods to be iterable
    store.list_feature_views.return_value = []
    store.list_entities.return_value = []
    store.list_data_sources.return_value = []
    return store


@pytest.fixture
def client(mock_feast_store, monkeypatch):
    """Create FeastClient with mock FeatureStore."""
    from kubeflow.feast.api.feast_client import FeastClient

    # Patch FeatureStore so __init__ uses the mock
    monkeypatch.setattr("feast.FeatureStore", lambda **kwargs: mock_feast_store)

    return FeastClient()


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="raises helpful ImportError when feast not installed",
            expected_status=FAILED,
            config={},
            expected_error=ImportError,
        ),
    ],
)
def test_init_import_error(test_case, monkeypatch):
    """Test that __init__ raises helpful ImportError when feast missing."""

    from kubeflow.feast.api.feast_client import FeastClient

    # Simulate missing feast by making import fail
    def mock_import(name, *args, **kwargs):
        if name == "feast":
            raise ImportError("No module named 'feast'")
        return __import__(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    try:
        FeastClient(**test_case.config)
        assert test_case.expected_status == SUCCESS
    except ImportError as e:
        assert test_case.expected_status == FAILED
        assert "pip install 'kubeflow[feast]'" in str(e)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default initialization with no args",
            expected_status=SUCCESS,
            config={},
            expected_output={"repo_path": None, "config": None},
        ),
        TestCase(
            name="initialization with repo_path",
            expected_status=SUCCESS,
            config={"repo_path": "/tmp/feast-repo"},
            expected_output={"repo_path": "/tmp/feast-repo", "config": None},
        ),
        TestCase(
            name="initialization with config dict",
            expected_status=SUCCESS,
            config={"config": {"project": "test"}},
            expected_output={"repo_path": None, "config": {"project": "test"}},
        ),
    ],
)
def test_init(test_case, monkeypatch):
    """Test FeastClient initialization with different configurations."""

    from kubeflow.feast.api.feast_client import FeastClient

    mock_feast_store_class = MagicMock()
    mock_feast_store_instance = MagicMock()
    mock_feast_store_class.return_value = mock_feast_store_instance

    monkeypatch.setattr("feast.FeatureStore", mock_feast_store_class)

    try:
        client = FeastClient(**test_case.config)

        assert test_case.expected_status == SUCCESS
        mock_feast_store_class.assert_called_once()
        assert client._store == mock_feast_store_instance
    except Exception as e:
        assert test_case.expected_status == FAILED
        if hasattr(test_case, "expected_error"):
            assert isinstance(e, test_case.expected_error)


def test_get_online_features(client, mock_feast_store):
    """Test get_online_features method."""
    # Setup mock return value
    mock_result = MagicMock()
    mock_result.to_dict.return_value = {
        "feature1": [1, 2, 3],
        "feature2": ["a", "b", "c"],
    }
    mock_feast_store.get_online_features.return_value = mock_result

    features = ["feature_view:feature1", "feature_view:feature2"]
    entity_rows = [{"entity_id": 1}, {"entity_id": 2}, {"entity_id": 3}]

    result = client.get_online_features(features=features, entity_rows=entity_rows)

    mock_feast_store.get_online_features.assert_called_once_with(
        features=features,
        entity_rows=entity_rows,
        full_feature_names=False,
    )
    assert result == {"feature1": [1, 2, 3], "feature2": ["a", "b", "c"]}


def test_list_feature_views(client, mock_feast_store):
    """Test list_feature_views method."""
    mock_feature_views = [MagicMock(name="fv1"), MagicMock(name="fv2")]
    mock_feast_store.list_feature_views.return_value = mock_feature_views

    result = client.list_feature_views()

    mock_feast_store.list_feature_views.assert_called_once()
    assert result == mock_feature_views


def test_list_entities(client, mock_feast_store):
    """Test list_entities method."""
    mock_entities = [MagicMock(name="entity1"), MagicMock(name="entity2")]
    mock_feast_store.list_entities.return_value = mock_entities

    result = client.list_entities()

    mock_feast_store.list_entities.assert_called_once()
    assert result == mock_entities


def test_list_data_sources(client, mock_feast_store):
    """Test list_data_sources method."""
    mock_data_sources = [MagicMock(name="ds1"), MagicMock(name="ds2")]
    mock_feast_store.list_data_sources.return_value = mock_data_sources

    result = client.list_data_sources()

    mock_feast_store.list_data_sources.assert_called_once()
    assert result == mock_data_sources


def test_apply(client, mock_feast_store):
    """Test apply method."""
    # Test with no objects
    client.apply()
    mock_feast_store.apply.assert_called_with([])

    # Test with objects
    mock_objects = [MagicMock(), MagicMock()]
    client.apply(objects=mock_objects)
    mock_feast_store.apply.assert_called_with(mock_objects)


def test_materialize(client, mock_feast_store):
    """Test materialize method."""
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()

    client.materialize(start_date=start_date, end_date=end_date)

    mock_feast_store.materialize.assert_called_once_with(
        start_date=start_date,
        end_date=end_date,
        feature_views=None,
    )


def test_materialize_incremental(client, mock_feast_store):
    """Test materialize_incremental method."""
    end_date = datetime.now()

    client.materialize_incremental(end_date=end_date)

    mock_feast_store.materialize_incremental.assert_called_once_with(
        end_date=end_date,
        feature_views=None,
    )


def test_store_property(client, mock_feast_store):
    """Test store property."""
    assert client.store == mock_feast_store


def test_feast_integration_with_local_setup():
    """Test Feast with a simple local setup.

    This test creates a minimal Feast feature store and validates basic operations.
    """
    pytest.importorskip("feast")
    pandas = pytest.importorskip("pandas")

    from feast.types import Float32, Int64

    from feast import Entity, FeatureView, Field, FileSource
    from kubeflow.feast.api.feast_client import FeastClient

    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "feature_repo"
        repo_path.mkdir()

        # Create a simple feature_store.yaml
        feature_store_yaml = repo_path / "feature_store.yaml"
        feature_store_yaml.write_text(
            """
project: test_project
provider: local
registry: data/registry.db
online_store:
    type: sqlite
    path: data/online_store.db
"""
        )

        # Create a data directory
        data_dir = repo_path / "data"
        data_dir.mkdir()

        # Create sample data
        sample_data = pandas.DataFrame(
            {
                "driver_id": [1001, 1002, 1003],
                "event_timestamp": [
                    datetime(2024, 1, 1, 12, 0, 0),
                    datetime(2024, 1, 1, 12, 0, 0),
                    datetime(2024, 1, 1, 12, 0, 0),
                ],
                "trips_today": [10, 15, 20],
                "rating": [4.5, 4.8, 4.2],
            }
        )
        sample_data_path = data_dir / "driver_stats.parquet"
        sample_data.to_parquet(sample_data_path)

        # Create feature definitions
        driver = Entity(name="driver", join_keys=["driver_id"])

        driver_stats_source = FileSource(
            name="driver_stats_source",
            path=str(sample_data_path),
            timestamp_field="event_timestamp",
        )

        driver_stats_fv = FeatureView(
            name="driver_stats",
            entities=[driver],
            schema=[
                Field(name="trips_today", dtype=Int64),
                Field(name="rating", dtype=Float32),
            ],
            source=driver_stats_source,
        )

        # Initialize client
        client = FeastClient(repo_path=str(repo_path))

        # Write feature definitions to store
        store = client.store
        store.apply([driver, driver_stats_source, driver_stats_fv])

        # Verify feature views
        feature_views = client.list_feature_views()
        assert len(feature_views) == 1
        assert feature_views[0].name == "driver_stats"

        # Verify entities
        entities = client.list_entities()
        assert len(entities) == 1
        assert entities[0].name == "driver"

        # Test online features retrieval after materialization
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 2, 0, 0, 0)
        client.materialize(start_date=start_date, end_date=end_date)

        # Get online features
        online_features = client.get_online_features(
            features=["driver_stats:trips_today", "driver_stats:rating"],
            entity_rows=[{"driver_id": 1001}, {"driver_id": 1002}],
        )

        assert "trips_today" in online_features
        assert "rating" in online_features
        assert len(online_features["trips_today"]) == 2

        print(f"✅ Feast integration test passed with {len(feature_views)} feature view(s)")
