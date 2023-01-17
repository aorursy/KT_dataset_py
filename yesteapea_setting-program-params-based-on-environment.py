import json

from os import path



import attr





def full_dir_path(relative_path: str) -> str:

    ret = path.abspath(path.expanduser(relative_path))

    if not ret.endswith('/'):

        ret = ret + '/'

    return ret





@attr.s

class Params(object):

    # Test mode runs quickly and produces horrible results!

    test_mode: float = attr.attrib(default=False)



    families_val_split: float = attr.attrib(default=0.1)

    input_data_dir: str = attr.attrib(default=full_dir_path('../input/'))

    output_data_dir: str = attr.attrib(default=full_dir_path('./'))

    # 1 +ve sample, 1 -ve sample.

    negative_sample_ratio: float = attr.attrib(default=0.5)

    # Batches of 32 (16 +ve, 16 -ve)

    batch_size: int = attr.attrib(default=32)

    # initiate_config overrides this in test_mode

    steps_per_epoch: int = attr.attrib(default=200)

    # initiate_config overrides this in test_mode

    validation_steps: int = attr.attrib(default=100)



    def override_params(self, config_file_path: str, verbose: bool = False):

        try:

            with open(config_file_path, "r") as f:

                config = json.load(f)

                if verbose:

                    print("Before initiate_config:", self)

                for k, v in config.items():

                    if k not in self.__dict__:

                        print("Ignoring invalid entry present in config. "

                              "entry key:", k)

                        pass

                    if k.endswith('_dir'):

                        v = full_dir_path(v)

                    self.__dict__[k] = v

            if verbose:

                print("After initiate_config:", self)

        except Exception as e:

            print("Exception while loading config json. This is okay when "

                  "running on kaggle. Error:", e)

        if self.test_mode:

            self.validation_steps = min(self.validation_steps, 5)

            self.steps_per_epoch = min(self.steps_per_epoch, 2)
params: Params = Params()

print("params:", params)
# Override params with a json config.

params.override_params('../input/program-params-demo-config/config.json')

print("The above override should succeed. Updated params: ", params)
# Update the params with invalid config. The update fails, but does not crash.

params.override_params('invalid_path.json')

print("The above override should fail. params: ", params)