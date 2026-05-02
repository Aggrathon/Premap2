import sys
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import yaml

if TYPE_CHECKING:
    import torch


class PremapInPath:
    """
    Add the src folder of the premap repo to the path temporarily.
    So that the local imports continue working (without dots).
    """

    def __init__(self, path: None | str = None):
        if path is not None:
            self.path = path
        else:
            self.path = str(Path(__file__).parent.parent / "premap")

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        sys.path.remove(self.path)


def construct_config(
    command_line: bool = False,
    post_config: None | Callable[[dict[str, dict[str, Any]]], None] = None,
    defaults: None | dict[str, object] = None,
    **kwargs,
):
    """Construct the `arguments.Config` object for `premap_main`.
    NOTE: This function assumes it is called from within a `with PremapInPath():`.

    Args:
        command_line: Also read commandline arguments.
        post_config: Optional post processing function that takes `arguments.Config`.
        defaults: Keyword arguments with lower priority than a config file.
        **kwargs: Keyword arguments with higher priority than commandline and config file (run `premap --help` for options).
    """
    import arguments  # type: ignore

    # Load default config.
    default_kwargs = vars(arguments.Config.defaults_parser.parse_args([]))
    if defaults is not None:
        default_kwargs.update(defaults)
    arguments.Config.construct_config_dict(default_kwargs)
    # Load command line args.
    if command_line:
        kwargs = vars(arguments.Config.no_defaults_parser.parse_args()) | kwargs
    # Read the yaml config files.
    if "config" in kwargs:
        with open(kwargs["config"], "r") as config_file:
            loaded_args = yaml.safe_load(config_file)
            arguments.Config.update_config_dict(arguments.Config.all_args, loaded_args)
    # Override with keyword args.
    arguments.Config.construct_config_dict(kwargs, nonexist_ok=False)
    if post_config is not None:
        post_config(arguments.Config)


def premap(
    model: "str | torch.nn.Module | None" = None,
    dataset: "str | tuple[torch.Tensor, int, float | torch.Tensor, float | torch.Tensor] | tuple[float | torch.Tensor, float | torch.Tensor] | None" = None,
    output_spec: "Literal['runnerup', 'verified-acc'] | torch.Tensor | None" = None,
    *,
    command_line: bool = False,
    post_config: None | Callable[[dict[str, dict[str, Any]]], None] = None,
    premap_path: None | str = None,
    defaults: dict[str, Any] | None = None,
    silent: bool = False,
    help: bool = False,
    **kwargs,
) -> list[Path] | list[BytesIO]:
    """Wrapper for PREMAP that takes keyword arguments (instead of commandline arguments).
    For keyword arguments run `get_arguments()` or `uv run premap --help` for options.

    Keyword Args:
        model: Neural network (name, string to eval, or torch module).
        dataset: Dataset (name, string to eval, `[X, labels, xmax, xmin]` or `[xmin, xmax]`).
        output_spec: Output specification ("runnerup": verify against the runnerup class, "verified/acc": verify against all other classes, or a tensor (multiplied with the output)).
        command_line: Also read commandline arguments.
        post_config: Optional post processing function that takes `arguments.Config`.
        premap_path: Path to the `src` folder of the PREMAP package.
        defaults: Keyword arguments with lower priority than a config file.
        silent: Do not print to stdout.
        help: Print command line instructions.
        **kwargs: Keyword arguments with higher priority than commandline and config file.

    Returns:
        List of result files (typically just one) that can be loaded with `torch.load`.
    """
    if help:
        return get_arguments(True, command_line)  # type: ignore
    with PremapInPath(premap_path):
        import preimage_main  # type: ignore

        if model is not None:
            kwargs["model"] = model
        if dataset is not None:
            kwargs["dataset"] = dataset
        if output_spec is not None:
            kwargs["robustness_type"] = output_spec

        construct_config(
            command_line=command_line,
            post_config=post_config,
            defaults=defaults,
            **kwargs,
        )
        if silent:
            with redirect_stdout(StringIO()):
                return preimage_main.main()
        else:
            return preimage_main.main()


def get_arguments(
    print: bool = False, command_line: bool = False
) -> None | list[tuple[str, type | list, str, Any]]:
    """List the available arguments for premap.
    See also `premap.arguments` and `uv run premap --help`.

    NOTE: Not all arguments are used for PREMAP, some are left over from αβ-CROWN.

    Args:
        print: If true, this function prints help information for the arguments instead of returning a list.
        command_line: When printing, print the same message as `uv run premap --help`.

    Returns:
        If `print==False` then a list of arguments as tuples with `(argument_name, choices_or_type, help_text, default_value)`.
    """
    with PremapInPath():
        import arguments  # type: ignore
        from torch import LongTensor, Tensor
        from torch.nn import Module

        if print and command_line:
            arguments.Config.defaults_parser.print_help()
        else:
            args = []
            for action in arguments.Config.defaults_parser._actions:
                if (
                    action.dest == "help"
                    or "deprecated" in action.help
                    or "o not use" in action.help
                ):
                    continue
                choice, default = action.type, action.default
                if action.choices is not None:
                    choice = action.choices
                elif action.dest == "model":
                    action.help = 'Model module or name (will be evaluated as a python statement). Also accepts \'Customized("file.py", "function")\'.'
                    if print:
                        choice = "str | Module"
                    else:
                        choice = str | Module
                elif action.dest == "dataset":
                    action.help = "Dataset tuple '(X, label, xmax, xmin)', '(xmin, xmax)', name (in 'utils.py'), or 'Customized(\"file.py\", \"function\")'."
                    if print:
                        choice = "str | tuple[Tensor, LongTensor, Tensor, Tensor]"
                    else:
                        choice = str | tuple[Tensor, LongTensor, Tensor, Tensor]
                elif action.dest == "robustness_type":
                    action.help = 'For robustness verification: verify against all labels ("verified-acc" mode), just the runnerup labels ("runnerup" mode), or with a custom linear function (Tensor).'
                    if print:
                        choice = '"verified-acc" | "runnerup" | Tensor'
                    else:
                        choice = Literal["verified-acc", "runnerup"] | Tensor
                elif action.dest == "log_prob":
                    if print:
                        choice = "str | Callable[[Tensor], Tensor]"
                    else:
                        choice = str | Callable[[Tensor], Tensor]
                elif action.type == arguments.keyvaluef:
                    choice = list[tuple[str, float]]
                elif action.type == arguments.str2bool or action.nargs == 0:
                    choice = bool
                if print:
                    if str(choice).startswith("<class"):
                        choice = choice.__qualname__
                    if isinstance(default, str):
                        default = f'"{action.default}"'
                args.append((action.dest, choice, action.help, default))
            if print:
                length = max(len(d) for d, *_ in args) - 5
                __builtins__["print"]("Available arguments to the premap function:")
                __builtins__["print"](
                    f"{'_Name':_<{length}}",
                    f"{'_Type':_<8}",
                    f"{'_Default':_<8}",
                    f"{'_Description':_<30}",
                    sep=" | ",
                )
                for name, typ, help, defa in args:
                    __builtins__["print"](
                        f"{name:<{length}}",
                        f"{str(typ):<8}",
                        f"{defa if defa else '':<8}",
                        help,
                        sep=" | ",
                    )
                __builtins__["print"](
                    " " * length, " " * 8, " " * 8, " " * 30, sep=" ^ "
                )
                __builtins__["print"](
                    "  Not all arguments are used for PREMAP, some are left over from αβ-CROWN."
                )
            else:
                return args


def cli():
    """Command line interface for PREMAP (reads arguments from `sys.argv`)."""
    if len(sys.argv) < 2:
        sys.argv.append("--help")
    premap(command_line=True)


if __name__ == "__main__":
    cli()
