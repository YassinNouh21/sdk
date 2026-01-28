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

from datetime import datetime
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
    """Create a mock FeatureStore."""
    store = MagicMock()
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
        assert client._feature_store == mock_feast_store_instance
    except Exception as e:
        assert test_case.expected_status == FAILED
        if hasattr(test_case, "expected_error"):
            assert isinstance(e, test_case.expected_error)


def test_feature_store_property(client, mock_feast_store):
    """Test feature_store property provides access to underlying FeatureStore."""
    assert client.feature_store == mock_feast_store


def test_feast_integration_with_local_setup():
    """Test Feast with a simple local setup.

    This test creates a minimal Feast feature store and validates basic operations
    through the feature_store property.
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

        # Use feature_store property to access full Feast functionality
        client.feature_store.apply([driver, driver_stats_source, driver_stats_fv])

        # Verify feature views
        feature_views = client.feature_store.list_feature_views()
        assert len(feature_views) == 1
        assert feature_views[0].name == "driver_stats"

        # Verify entities
        entities = client.feature_store.list_entities()
        assert len(entities) == 1
        assert entities[0].name == "driver"

        # Test online features retrieval after materialization
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 2, 0, 0, 0)
        client.feature_store.materialize(start_date=start_date, end_date=end_date)

        # Get online features using feature_store
        online_features = client.feature_store.get_online_features(
            features=["driver_stats:trips_today", "driver_stats:rating"],
            entity_rows=[{"driver_id": 1001}, {"driver_id": 1002}],
        ).to_dict()

        assert "trips_today" in online_features
        assert "rating" in online_features
        assert len(online_features["trips_today"]) == 2

        print(f"✅ Feast integration test passed with {len(feature_views)} feature view(s)")
