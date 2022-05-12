import ast
import argparse
from src.utils import get_logger
import yaml

log = get_logger(__name__)


def main():
    # Generate template of configuration files for a new model.

    parser = argparse.ArgumentParser(description='Build default config for a specified model + datamodule')
    parser.add_argument('-m', '--model', type=str, help='model file')
    parser.add_argument('-d', '--datamodule', type=str, help='datamodule file')
    parser.add_argument('-n', '--name', type=str, help='name of the model')

    args = parser.parse_args()

    model_config = {}
    recursively_update_class_arguments(model_config, args.model.split('.')[0])

    datamodule_config = {}
    recursively_update_class_arguments(datamodule_config, args.datamodule.split('.')[0])

    with open(f'configs/model/{args.name}.yml', 'w') as outfile:
        yaml.dump(model_config, outfile, default_flow_style=False)

    with open(f'configs/datamodule/{args.name}.yml', 'w') as outfile:
        yaml.dump(datamodule_config, outfile, default_flow_style=False)




def recursively_update_class_arguments(argument_dict, module, class_name=None, prefix='', add_target=True):
    with open(prefix + module.replace('.', '/') + '.py', "r") as source:
        tree = ast.parse(source.read())

    imports_from = [n for n in tree.body if isinstance(n, ast.ImportFrom)]
    import_dict = {name.name: imp.module for imp in imports_from for name in imp.names}
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]

    if class_name is None:
        log.info('Inferring class name (using first class found)')
        class_name = classes[0].name

    class_ = next((cls for cls in classes if cls.name == class_name))
    
    if class_ is not None:
        if add_target:
            argument_dict['_target_'] = f'{module}.{class_name}'
        init = next((n for n in class_.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'))
        if init is None:
            return
        args = init.args.args[1:]  # drop self
        defaults = ['???']*(len(args)-len(init.args.defaults)) + [ast.literal_eval(df) for df in init.args.defaults]
        annotations = [arg.annotation.id if isinstance(arg.annotation, ast.Name) else None for arg in args]
        
        for arg, dft, ann in zip(args, defaults, annotations):
            if dft == '???' and ann is not None and import_dict.get(ann) is not None:
                argument_dict[arg.arg] = {}
                recursively_update_class_arguments(argument_dict[arg.arg], import_dict[ann], ann)
            else:
                argument_dict[arg.arg] = dft


if __name__ == "__main__":
    main()
