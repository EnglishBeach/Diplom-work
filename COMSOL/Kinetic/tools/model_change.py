import mph


def copy_settings(from_node: mph.Node, to_node: mph.Node):
    for i in range(2):
        from_properties, to_properties = from_node.properties(), to_node.properties()

        auto_settings = {}
        for key, value in from_properties.items():
            if (str(value) == 'auto') or (to_properties.get(key, None) is None):
                auto_settings.update({key: value})
                continue
            to_node.property(name=key, value=value)

        for key, value in auto_settings.items():
            if to_properties.get(key, None) is None:
                continue
            to_node.property(name=key, value=value)


def compare(from_node: mph.Node, to_node: mph.Node):
    from_properties, to_properties = from_node.properties(), to_node.properties()
    all_properties = sorted(set(from_properties).union(set(to_properties)))

    print(f"{'from ' + from_node.name(): >53} | {'to '+to_node.name(): <53}")
    for key in all_properties:
        from_prop = str(from_properties.get(key, None))
        to_prop = str(to_properties.get(key, None))

        string = (
            f'{"*" if (from_prop!=to_prop) else " "} {key: <30} '
            + f'{from_prop[:20]: <20} | {to_prop[:20]: <20}'
        )
        print(string)


def copy_solver(from_solver: mph.Node, to_solver: mph.Node, verbose=False):
    copy_settings(from_solver, to_solver)
    from_dict = {node.name(): node for node in from_solver.children()}
    to_dict = {node.name(): node for node in to_solver.children()}

    for node in from_dict:
        from_node = from_dict[node]
        to_node = to_dict[node]
        copy_settings(from_node, to_node)
        if verbose:
            compare(from_node, to_node)
            print('*' * 120)
