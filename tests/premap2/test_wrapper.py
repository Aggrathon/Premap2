from premap2.wrapper import construct_config, PremapInPath


def test_construct_config():
    def post(config):
        assert config["debug"]["asserts"]

    with PremapInPath():
        construct_config(False, post, {"asserts": True})
        construct_config(False, post, asserts=True)
        construct_config(False, post, {"asserts": False}, asserts=True)
