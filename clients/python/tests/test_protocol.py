import json
import socket
import struct
import threading
import unittest

from ertw_client.protocol import (
    ACTION_STRIDE,
    FRAME_ACTION,
    FRAME_LIFECYCLE,
    FRAME_OBSERVATION,
    HEADER_BYTES,
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    WIRE_MAGIC,
    Action,
    Frame,
    FrameHeader,
    ProtocolError,
    decode_json,
    decode_observation,
    encode_action,
    encode_json_frame,
    read_frame,
    validate_extension,
    validate_lifecycle,
    validate_metadata,
)


def valid_metadata():
    return {
        "protocol_version": 4,
        "schema_version": 1,
        "fixed_timestep_seconds": 1.0 / 60.0,
        "physics_ticks_per_decision": 1,
        "observation_floats": 47,
        "action_floats": 7,
        "self_stride": 8,
        "neighbor_stride": 15,
        "field_count": 3,
        "max_neighbors": 2,
        "field_samples": 1,
        "field_channels": 3,
        "sensor_radius": 8.0,
        "action_min": [0.0] * 7,
        "action_max": [1.0] * 7,
        "action_semantics": ["continuous"] * 7,
        "transport_mode": "lockstep",
        "world_seed": 9,
        "world_tick": 0,
        "world_id": 10,
        "session_id": 11,
        "resume_token": "secret",
        "stable_agent_id": 12,
        "snapshot_schema_version": 2,
        "capabilities": ["lockstep"],
    }


class ProtocolTests(unittest.TestCase):
    def test_header_preserves_full_width_identifiers(self):
        header = FrameHeader(
            version=PROTOCOL_VERSION,
            kind=FRAME_ACTION,
            frame_bytes=HEADER_BYTES + ACTION_STRIDE * 4,
            step=0x0123_4567_89AB_CDEF,
            entity_id=0xFEDC_BA98_7654_3210,
            payload_floats=ACTION_STRIDE,
        )
        self.assertEqual(FrameHeader.from_bytes(header.to_bytes()), header)

    def test_action_layout_matches_wire_contract(self):
        action = Action(0.25, -0.5, 0.75, 1.0, 0.0, 3.0, 1.5)
        encoded = encode_action(9, 11, action)
        words = struct.unpack("<14I", encoded[:HEADER_BYTES])
        self.assertEqual(words[0], WIRE_MAGIC)
        self.assertEqual(words[2], FRAME_ACTION)
        self.assertEqual(words[3], HEADER_BYTES + ACTION_STRIDE * 4)
        self.assertEqual(words[4], 9)
        self.assertEqual(words[6], 11)
        self.assertEqual(words[12], ACTION_STRIDE)
        self.assertEqual(struct.unpack("<7f", encoded[HEADER_BYTES:]), action.values())

    def test_observation_decodes_relation_mask_and_valid_count(self):
        values = [0.0] * (8 + 3 * 1 * 3 + 15)
        neighbor = 17
        values[neighbor : neighbor + 15] = [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            0x1234,
            0x5678,
            0x9ABC,
            0xDEF0,
            0.25,
            2.5,
            1.25,
            1.0,
        ]
        payload = struct.pack(f"<{len(values)}f", *values)
        frame = Frame(
            FrameHeader(
                version=PROTOCOL_VERSION,
                kind=FRAME_OBSERVATION,
                frame_bytes=HEADER_BYTES + len(payload),
                step=13,
                entity_id=17,
                max_neighbors=1,
                neighbor_count=1,
                field_samples=1,
                field_channels=3,
                payload_floats=len(values),
            ),
            payload,
        )
        observation = decode_observation(frame)
        self.assertEqual(observation.step, 13)
        self.assertEqual(observation.neighbors[0].tags, 0xDEF0_9ABC_5678_1234)
        self.assertTrue(observation.neighbors[0].valid)

    def test_json_frame_round_trips_through_fragmented_socket_reads(self):
        left, right = socket.socketpair()
        encoded = encode_json_frame(FRAME_LIFECYCLE, 5, 7, {"kind": "entity_alive"})

        def send_fragments():
            for offset in range(0, len(encoded), 3):
                left.sendall(encoded[offset : offset + 3])
            left.close()

        sender = threading.Thread(target=send_fragments)
        sender.start()
        frame = read_frame(right)
        sender.join()
        right.close()
        self.assertEqual(decode_json(frame), {"kind": "entity_alive"})

    def test_rejects_reserved_word_and_oversized_frame(self):
        words = [WIRE_MAGIC, PROTOCOL_VERSION, FRAME_LIFECYCLE, HEADER_BYTES]
        words.extend([0] * 10)
        words[13] = 1
        with self.assertRaisesRegex(ProtocolError, "reserved"):
            FrameHeader.from_bytes(struct.pack("<14I", *words))
        words[13] = 0
        words[3] = MAX_FRAME_BYTES + 1
        with self.assertRaisesRegex(ProtocolError, "frame length"):
            FrameHeader.from_bytes(struct.pack("<14I", *words))

    def test_rejects_non_object_json(self):
        payload = json.dumps(["not", "an", "object"]).encode()
        frame = Frame(
            FrameHeader(
                version=PROTOCOL_VERSION,
                kind=FRAME_LIFECYCLE,
                frame_bytes=HEADER_BYTES + len(payload),
            ),
            payload,
        )
        with self.assertRaisesRegex(ProtocolError, "object"):
            decode_json(frame)

    def test_rejects_nonfinite_actions(self):
        with self.assertRaisesRegex(ProtocolError, "finite"):
            encode_action(0, 0, Action(force_x=float("nan")))

    def test_metadata_dimensions_are_cross_checked(self):
        metadata = valid_metadata()
        validate_metadata(metadata)
        metadata["observation_floats"] = 46
        with self.assertRaisesRegex(ProtocolError, "observation length"):
            validate_metadata(metadata)

    def test_metadata_rejects_coerced_integer_fields(self):
        metadata = valid_metadata()
        metadata["max_neighbors"] = "2"
        with self.assertRaisesRegex(ProtocolError, "must be an integer"):
            validate_metadata(metadata)

    def test_lifecycle_and_extension_payloads_are_validated(self):
        lifecycle = {
            "sequence": 1,
            "world_tick": 2,
            "kind": "session_attached",
            "subject_id": 3,
            "related_id": None,
            "lineage_id": None,
            "generation": None,
            "reason": "connected",
        }
        validate_lifecycle(lifecycle)
        lifecycle["sequence"] = 0
        with self.assertRaisesRegex(ProtocolError, "out of range"):
            validate_lifecycle(lifecycle)

        validate_extension({"decision_sequence": 1, "delta": None})
        with self.assertRaisesRegex(ProtocolError, "finite"):
            validate_extension(
                {
                    "decision_sequence": 1,
                    "delta": {"energy": float("nan"), "structure": 0.0, "mass": 0.0},
                }
            )


if __name__ == "__main__":
    unittest.main()
