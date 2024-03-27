import spu

alice_ip = '127.0.0.1'
bob_ip = '127.0.0.1'

# SPU settings
cluster_def = {
    'nodes': [
        {'party': 'alice', 'id': 'local:0', 'address': alice_ip + ':12945'},
        {'party': 'bob', 'id': 'local:1', 'address': bob_ip + ':12946'},
        # {'party': 'carol', 'id': 'local:2', 'address': '127.0.0.1:12347'},
    ],
    'runtime_config': {
        # SEMI2K support 2/3 PC, ABY3 only support 3PC, CHEETAH only support 2PC.
        # pls pay attention to size of nodes above. nodes size need match to PC setting.
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
    },
}

# HEU settings
heu_config = {
    'sk_keeper': {'party': 'alice'},
    'evaluators': [{'party': 'bob'}],
    'mode': 'PHEU',
    'he_parameters': {
        # ou is a fast encryption schema that is as secure as paillier.
        'schema': 'paillier',
        'key_pair': {
            'generate': {
                # bit size should be 2048 to provide sufficient security.
                'bit_size': 2048,
            },
        },
    },
    'encoding': {
        'cleartext_type': 'DT_I32',
        'encoder': "IntegerEncoder",
        'encoder_args': {"scale": 1},
    },
}