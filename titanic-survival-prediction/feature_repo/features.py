from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Int64, Float32, String

# 1. Define the Entity (Primary Key)
passenger = Entity(name="passenger", join_keys=["PassengerId"])

# 2. Define the Source (The Parquet file we just made)
titanic_source = FileSource(
    path="data/titanic_features.parquet",
    timestamp_field="event_timestamp"
)

# 3. Define the Feature View
# This groups our features together so we can retrieve them.
titanic_view = FeatureView(
    name="titanic_passenger_stats",
    entities=[passenger],
    ttl=timedelta(days=365),  # How far back to look - 1 year
    schema=[
        Field(name="Pclass", dtype=Int64),
        Field(name="Age", dtype=Float32),
        Field(name="FamilySize", dtype=Int64),
        Field(name="Fare", dtype=Float32),
        Field(name="Sex", dtype=String),
        Field(name="Title", dtype=String),
        Field(name="Embarked", dtype=String),
        Field(name="CabinLetter", dtype=String),
        Field(name="TicketPrefix", dtype=String)
    ],
    online=True, # Enable this for online serving
    source=titanic_source,
)