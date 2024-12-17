import decimal

import numpy as np
from boto3.dynamodb.types import BOOLEAN, NUMBER, STRING
from boto3.dynamodb.types import TypeDeserializer as _TypeDeserializer
from boto3.dynamodb.types import TypeSerializer as _TypeSerializer


class TypeSerializer(_TypeSerializer):
    def __init__(self, precision=8):
        super().__init__()
        self.precision = precision

    def _get_dynamodb_type_and_value(self, value):
        dynamodb_type = None

        if self._is_float(value):
            if np.isfinite(value):
                dynamodb_type = NUMBER
                value = decimal.Decimal(f"{value:0.{self.precision}g}")
            else:
                dynamodb_type = STRING
                if np.isposinf(value):
                    value = "float:+Infinity"
                elif np.isneginf(value):
                    value = "float:-Infinity"
                elif np.isnan(value):
                    value = "float:NaN"
                else:
                    raise TypeError("Unknown non-finite float not supported")
        if self._is_np_int(value):
            dynamodb_type = NUMBER
            value = decimal.Decimal(f"{value:d}")
        if self._is_np_bool(value):
            dynamodb_type = BOOLEAN
            value = bool(value)
        else:
            dynamodb_type = super()._get_dynamodb_type(value)

        return dynamodb_type, value

    def _is_float(self, value):
        if isinstance(value, (float, np.floating)):
            return True
        return False

    def _is_np_int(self, value):
        if isinstance(value, np.integer):
            return True
        return False

    def _is_np_bool(self, value):
        if isinstance(value, np.bool_):
            return True
        return False

    def serialize(self, value):
        """
        The method to serialize the Python data types.

        See base class for additional details

        Parameters
        ----------
        value : Any
            A python value to be serialized to DynamoDB.

        Returns
        -------
        dict
            A dictionary that represents a dynamoDB data type.
        """
        dynamodb_type, value = self._get_dynamodb_type_and_value(value)
        serializer = getattr(self, "_serialize_%s" % dynamodb_type.lower())
        return {dynamodb_type: serializer(value)}


class TypeDeserializer(_TypeDeserializer):
    def _deserialize_s(self, value):
        if value == "float:+Infinity":
            return np.inf
        elif value == "float:-Infinity":
            return -np.inf
        elif value == "float:NaN":
            return np.nan
        else:
            return value

    def _deserialize_n(self, value):
        value = super()._deserialize_n(value)
        if value == int(value):
            return int(value)
        else:
            return float(value)

    def _deserialize_b(self, value):
        return bytes(value)
