#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from collections import defaultdict
from pyverilog.vparser.parser import parse
from pyverilog.vparser.ast import InstanceList, Instance, Decl, Assign, Wire as WireNode, PortArg, Portlist, Identifier, Input, Pointer, Output, \
    ModuleDef, Pragma, Port, IntConst
from copy import copy
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator

parser = argparse.ArgumentParser(description="component splitter.")
parser.add_argument("--src", "-s", dest="src_file", help="input file")
parser.add_argument("--dest", "-d", dest="result_file", help="output file")

args = parser.parse_args()

def describe_node(node):
    print((type(node), node.__class__.__name__, node.attr_names, node.children(), node.description, node.lineno, node.name))


class VWire(object):
    def __init__(self, decl):
        self.name = decl.name

        self.wire = decl
        self.input = None
        self.outputs = []
        self.wires = {}
        # 如果decl的width存在，则是总线
        # if decl.width:
        #     for i in range(int(decl.width.lsb.value), int(decl.width.msb.value)+1):
        #         self.wires[i] = VWire(self.create_wire(self.name+'_'+str(i)))
        #         self.wires[i] = VWire(self.create_wire(self.name+'_'+str(i)))

    def __str__(self):
        return self.name

    def create_wire(self, name):
        return WireNode(name=name)

    def add_port(self, instance, port):
        if isinstance(instance, Instance):
            if instance.module.startswith("and") or instance.module.startswith("or"):
                if port.portname in ("a", "b"):
                    self.outputs.append((instance, port))
                else:
                    if self.input:
                        raise Exception("invalid situation, wire's input already existed")
                    self.input = ((instance, port))
            elif instance.module.startswith("spl"):
                if port.portname == "a":
                    self.outputs.append((instance, port))
                else:
                    if self.input:
                        raise Exception("invalid situation, wire's input already existed, %s, input: %s", self.name, self.input[0].name)
                    self.input = ((instance, port))
            elif port.portname == "din":
                self.outputs.append((instance, port))
            elif instance.module.startswith("dff"):
                # TODO: dff的进出口分别是什么?
                if port.portname in ("CLK", "D"):
                    self.outputs.append((instance, port))
                else:
                    self.input = ((instance, port),)
            else:
                if self.input:
                    raise Exception("invalid situation, wire's input already existed, %s", self.name)
                self.input = ((instance, port))
        elif isinstance(instance, Pointer):
            self.outputs.append((instance, None))

def toggle_interface_name(interface_name):
    if interface_name == "b":
        return "i"
    else:
        return "b"

def reverse_instance_name(instance, port):
    instance_name_prefix, input1, input2 = instance.module[:-2], instance.module[-2], instance.module[-1]
    if port.portname == "a":
        return "".join([instance_name_prefix, toggle_interface_name(input1), input2])
    else:
        return "".join([instance_name_prefix, input1, toggle_interface_name(input2)])

class NumberMapping(object):
    start = 10000
    def birth(self):
        self.start += 1
        return self.start

number_mapping = NumberMapping()

class Splitter(object):
    splt_dict = {}
    def __init__(self, in_wire, dout, reserve_interface=False):
        """dout is a list, with 4 output at most"""
        self.port_to_output = {}
        cnt = len(dout)
        if reserve_interface:
            cnt += 1
        self.wires = self.create_wires(cnt)
        portlist = []
        # self.out_interfaces = Portlist()
        out_interfaces_name = ['b', 'c', 'd', 'e']
        for index in range(cnt):
            i = out_interfaces_name[index]
            portarg = PortArg(portname=i, argname=Identifier(self.wires[index].name))
            setattr(self, i, portarg)
            portlist.append(portarg)
            if index == (cnt-1) and reserve_interface:
                pass
            else:
                # TODO modify the output related components
                # dout[index].wire = splitter_out_interface.wire
                if isinstance(dout[index], tuple) and isinstance(dout[index][0], Instance):
                    dout[index][1].argname = portarg.argname
                elif isinstance(dout[index], tuple) and isinstance(dout[index][0], (Output, Pointer)):
                    self.port_to_output[portarg] = dout[index][0]
                elif isinstance(dout[index], (Output, Pointer)):
                    self.port_to_output[portarg] = dout[index]
                else:
                    raise Exception("unexcepted dout type, %s", dout)

        self.out_interfaces = portlist
        self.out_ports = Portlist(ports=portlist)
        self.split_cnt = len(portlist)
        if isinstance(in_wire, Pointer):
            self.a = PortArg(portname="a", argname=in_wire)
        elif isinstance(in_wire.name, str):
            self.a = PortArg(portname="a", argname=Identifier(in_wire.name))
        portlist.insert(0, self.a)
        self.ports = Portlist(ports=portlist)

        self.number = self.birth()
        self.name = '_' + str(self.number) + '_'
        self.child = None

    def __str__(self):
        return "%d: in %s" % (self.number, str(self.din))

    @property
    def format_node_name(self):
        return "spl" + str(self.split_cnt)

    def to_ast_node(self):
        insts = []
        if self.child:
            insts.extend(self.child.to_ast_node())
        inst = Instance(self.format_node_name, self.name, portlist=self.ports.ports, parameterlist=[])
        inst_list = InstanceList(self.format_node_name, parameterlist=[], instances=[inst])
        insts.append(inst_list)
        return insts

    def collect_port_to_output(self):
        links = copy(self.port_to_output)
        if self.child:
            links.update(self.child.collect_port_to_output())
            return links
        else:
            return self.port_to_output

    def collect_wires(self):
        wires = [wire for wire in self.wires]
        if self.child:
            wires.extend(self.child.collect_wires())
        return wires

    def wires_to_ast_node(self):
        wires = self.collect_wires()
        wires_ast_node = []
        for wire in wires:
            wires_ast_node.append(Decl(list=[wire]))
        return wires_ast_node

    def add_child(self, child):
        self.child = child

    @classmethod
    def create_wires(cls, cnt):
        wires = []
        for i in range(cnt):
            new_number = number_mapping.birth()
            _name_ = '_' + str(new_number) + '_'
            wires.append(WireNode(name=_name_))
        return wires

    @classmethod
    def birth(cls):
        return number_mapping.birth()

    @property
    def dout(self):
        return [x for x in (self.b, self.c, self.d, self.e) if x]

    @property
    def din(self):
        return self.a

    @din.setter
    def din(self, value):
        self.a = value

    @property
    def last_interface(self):
        return self.out_interfaces[-1]

    @property
    def next_splitter_wire(self):
        return self.wires[-1]

    def to_text(self):
        valid_output = self.out_interfaces
        output_cnt = len(valid_output)
        keyword = "spl"+str(output_cnt)
        _id_ = "_" + str(self.number) + "_"
        a = ".a(%s)" % self.a.wire
        output_list = []
        for x in ("b", "c", "d", "e"):
            xattr = getattr(self, x, None)
            if xattr:
                output_list.append(".%s(%s)" % (x, str(xattr.wire)))
        outputs = ",\n\t".join(output_list)
        return "%s %s (\n\t%s,\n\t%s\n)" % (keyword, _id_, a, outputs)

class Node(object):
    reverse_mapping = {}
    def __init__(self, name, ast_node=None, ptr=None, original_name=None):
        self.name = name
        self.children = []
        self.parent = []
        self.depth = None
        self._ptr = ptr
        self.original_name = original_name
        self.ast_node = ast_node

    def __str__(self):
        return self.name

    @property
    def ast_node(self):
        return self._ast_node

    @ast_node.setter
    def ast_node(self, value):
        self._ast_node = value
        if isinstance(self._ast_node, Instance):
            self.reverse_mapping[self._ast_node.name] = self
        elif isinstance(self._ast_node, (Input, Output)):
            if self._ptr is None:
                self.reverse_mapping[self._ast_node.name] = self
            else:
                self.reverse_mapping[self._ast_node.name + "_" + str(self._ptr)] = self

    @classmethod
    def reset_reverse_mapping(cls):
        cls.reverse_mapping = {}

    def add_child(self, child, port):
        self.children.append((child, port))

class NodeTree(object):
    extra_ports = []
    input_decl_starts_lineno = 0
    output_decl_starts_lineno = 0
    def __init__(self, ast):
        self.count = 0
        self.ast = ast
        self.init_empty_values()
        self.parse_ast()
        # self.remove_inv()
        self.add_splitter()
        # self.set_module_def_to_ast()
        self.reparse(self.ast)

    def order(self):
        items = []
        decl = []
        instances = []
        assigns = []
        for item in self.module.items:
            if isinstance(item, Decl):
                decl.append(item)
            elif isinstance(item, InstanceList):
                instances.append(item)
            elif isinstance(item, Assign):
                assigns.append(item)
            elif isinstance(item, Pragma):
                pass
            else:
                raise Exception("item type not excepted")

        items.extend(decl)
        items.extend(instances)
        items.extend(assigns)
        self.module.items = items

    def add_extra_wires(self, wire):
        wire_node = []
        for i in range(int(wire.width.lsb.value), int(wire.width.msb.value)+1):
            wire_name = wire.name+'_'+str(i)
            twire = WireNode(name=wire_name)
            vwire = VWire(twire)
            self.wires[wire_name] = vwire
            wire_node.append(twire)
        return wire_node

    def add_extra_wires_in_ast(self, wires, items):
        module_def = self.ast.description.definitions[2]
        for wire in wires:
            items.append(Decl(list=(wire,)))
        module_def.items = items
        definitions = list(self.ast.description.definitions)
        definitions[2] = module_def
        self.ast.description.definitions = tuple(definitions)
        return module_def.items

    def add_extra_port(self, wire):
        for i in range(int(wire.width.lsb.value), int(wire.width.msb.value)+1):
            port_name = wire.name+'_'+str(i)
            self.extra_ports.append(port_name)
            self.add_definition_ports(port_name)

    def add_definition_ports(self, name):
        module_def = self.ast.description.definitions[2]
        ports = list(module_def.portlist.ports)
        ports.append(Port(name=name, width=None, type=None))
        module_def.portlist.ports = tuple(ports)
        definitions = list(self.ast.description.definitions)
        definitions[2] = module_def
        self.ast.description.definitions = tuple(definitions)

    def init_empty_values(self):
        self.input_nodes = {}
        self.output_nodes = {}
        self.wires = {}
        self.instances = {}
        self.nodes = {}
        self.buffer_cnt = 0
        self.output_to_assign = {}
        self.module = None
        for module in self.ast.description.definitions:
            if isinstance(module, ModuleDef):
                self.module = module
        if not self.module:
            raise Exception("no valid module found")
        self.module.items = list(self.module.items)

    def set_module_def_to_ast(self):
        definitions = []
        for module in self.ast.description.definitions:
            if not isinstance(module, ModuleDef):
                definitions.append(module)
            else:
                definitions.append(self.module)
                break
        self.ast.description.definitions = definitions

    def reparse(self, ast):
        Node.reset_reverse_mapping()
        self.init_empty_values()
        self.parse_ast()

    def generate_splitter(self, wire, douts):
        is_wire = wire.name in self.wires
        if not is_wire and wire.name in self.input_nodes and wire._ptr is not None:
            wire = Pointer(Identifier(wire.original_name), IntConst(wire._ptr))

        if len(douts) > 4:
            first_spiltter = Splitter(wire, douts[:3], reserve_interface=True)
            child_splitter = self.generate_splitter(first_spiltter.next_splitter_wire, douts[3:])
            first_spiltter.add_child(child_splitter)
            if is_wire:
                wire.outputs = [first_spiltter]
            return first_spiltter
        else:
            splitter = Splitter(wire, douts)
            if is_wire:
                wire.outputs = [splitter]
            return splitter

    def add_splitter(self):
        for _, wire in list(self.wires.items()):
            if len(wire.outputs) > 1:
                douts = copy(wire.outputs)
                splitter = self.generate_splitter(wire, douts)
                self.module.items.extend(splitter.to_ast_node())
                self.module.items.extend(splitter.wires_to_ast_node())
                self.update_splitter_to_assign(splitter.collect_port_to_output())
        for _, input_node in list(self.input_nodes.items()):
            # print(_, input_node, len(input_node.children))
            if len(input_node.children) > 1:
                douts = copy(input_node.children)
                splitter = self.generate_splitter(input_node, douts)
                self.module.items.extend(splitter.to_ast_node())
                self.module.items.extend(splitter.wires_to_ast_node())
                self.update_splitter_to_assign(splitter.collect_port_to_output())

    def update_splitter_to_assign(self, ports_to_output):
        for port, output in list(ports_to_output.items()):
            if isinstance(output, Pointer):
                assign_item = self.output_to_assign[output]
            else:
                assign_item = self.output_to_assign[output.name]

            assign_item.right.var.name = port.argname.name

    def remove_inv(self):
        items = self.module.items
        invertors = []
        for instance in list(self.instances.values()):
            if instance.module == "inv":
                in_wire = instance.portlist[0].argname
                out_wire = instance.portlist[1].argname
                if isinstance(out_wire, Pointer):
                    out_wire_name = str(out_wire.var) + '_' + str(out_wire.ptr)
                    if out_wire_name not in self.wires:
                        continue
                if out_wire.name in self.wires:
                    ports = [port for output, port in self.wires[out_wire.name].outputs]
                else:
                    continue
                if not all(ports):
                    continue

                invertors.append(instance)
                # 对本wire接入的每个器件,都直接把inv的入口线接入,然后修改他们对应口的bi属性
                # 同时更新wire的属性,将invertor的出口线从wires集合中移除,并将原出口线的关联instances
                # 复制到入口线去
                if isinstance(in_wire, Pointer):
                    wire_name = str(in_wire.var) + '_' + str(in_wire.ptr)
                    if wire_name in self.wires:
                        self.wires[wire_name].outputs = [(x, y) for x, y in self.wires[wire_name].outputs if x != instance]
                        #  如果不在wires里，则invertor的入口可能是输入端口
                else:
                    self.wires[in_wire.name].outputs = [(x, y) for x, y in self.wires[in_wire.name].outputs if x != instance]
                for output, port in self.wires[out_wire.name].outputs:
                    if isinstance(in_wire, Pointer):
                        in_wire_name = str(in_wire.var) + '_' + str(in_wire.ptr)
                    else:
                        in_wire_name = in_wire.name
                    if in_wire_name in self.wires:
                        self.wires[in_wire_name].outputs.append((output, port))
                    if port:
                        port.argname = in_wire
                        output.module = reverse_instance_name(output, port)
                        self.instances[output.name].module = output.module
                    else:
                        # 如果下一个器件直接是输出端，应该修改一个assign语句
                        pass
                del self.wires[out_wire.name]
        items = [x for x in items if not isinstance(x, InstanceList) or x.instances[0] not in invertors]
        self.module.items = items

    def parse_decl(self, items):
        items_to_be_delete = []
        wires_to_be_add = []
        for i, item in enumerate(items):
            if isinstance(item, Decl):
                decl_item = item.list[0]

                if isinstance(decl_item, WireNode):
                    # print decl_item.name, decl_item.width.lsb, decl_item.width.msb
                    wire = decl_item
                    if wire.width:
                        items_to_be_delete.append(item)
                        wires_to_be_add.extend(self.add_extra_wires(wire))
                        # self.add_extra_port(wire)
                    else:
                        vwire = VWire(decl=wire)
                        self.wires[wire.name] = vwire
                        if vwire.wires:
                            for child_wire in list(vwire.wires.values()):
                                self.wires[child_wire.name] = child_wire
                elif isinstance(decl_item, Input):
                    if not self.input_decl_starts_lineno:
                        self.input_decl_starts_lineno = decl_item.lineno
                    if decl_item.width:
                        msb = int(decl_item.width.msb.value) + 1
                        lsb = int(decl_item.width.lsb.value)
                        for i in range(lsb, msb):
                            node_name = decl_item.name + '_' + str(i)
                            self.input_nodes[node_name] = Node(name=node_name, ast_node=decl_item, ptr=i, original_name=decl_item.name)
                    else:
                        self.input_nodes[decl_item.name] = Node(name=decl_item.name, ast_node=decl_item)
                elif isinstance(decl_item, Output):
                    if not self.output_decl_starts_lineno:
                        self.output_decl_starts_lineno = decl_item.lineno
                    if decl_item.width:
                        msb = int(decl_item.width.msb.value) + 1
                        lsb = int(decl_item.width.lsb.value)
                        for i in range(lsb, msb):
                            node_name = decl_item.name + '_' + str(i)
                            self.output_nodes[node_name] = Node(name=node_name, ast_node=decl_item, ptr=i, original_name=decl_item.name)
                    else:
                        self.output_nodes[decl_item.name] = Node(name=decl_item.name, ast_node=decl_item)
        items = [x for x in items if x not in items_to_be_delete]
        items = self.add_extra_wires_in_ast(wires_to_be_add, items)
        return items

    def add_output_decl(self, node_name):
        items = self.module.items
        decl_item = Output(name=node_name)
        decl_item_list = [decl_item]
        output_decl = Decl(list=decl_item_list)
        items.insert(self.output_decl_starts_lineno+1, output_decl)
        self.module.items = items
        self.output_nodes[decl_item.name] = Node(name=decl_item.name, ast_node=decl_item)

    def add_input_decl(self, node_name):
        items = self.module.items
        decl_item = Input(name=node_name)
        decl_item_list = [decl_item]
        input_decl = Decl(list=decl_item_list)
        items.insert(self.input_decl_starts_lineno+1, input_decl)
        self.module.items = items
        self.input_nodes[decl_item.name] = Node(name=decl_item.name, ast_node=decl_item)

    def change_instance_port_wire(self, instance, port, new_wire):
        """
        :param instance: Instance
        :param port: Port
        :param new_wire: Wire Identifier
        :return:
        """
        for p in instance.portlist:
            if p.portname == port.portname:
                p.argname = Identifier(name=new_wire.name)
                break
        return instance

    def parse_assign(self, items):
        assign_to_remove = []
        for i, item in enumerate(items):
            if isinstance(item, Assign):
                if isinstance(item.right.var, Identifier):
                    if isinstance(item.left.var, Pointer):
                        # 如果左值是Pointer也就是input/output类型, 且右值为identifier,则认为是将wire赋值给decoder结束节点
                        # 如果左值是Pointer类型,查看wire里是否有对应的值,有的话将对应的wire的outputs置为右值的outputs,同时删除左值
                        node_name = item.left.var.var.name + "_" + str(item.left.var.ptr.value)
                        right_var_name = item.right.var.name

                        if node_name in self.wires and right_var_name in self.wires:
                            left_wire = self.wires[node_name]
                            right_wire = self.wires[right_var_name]
                            ins, _port = left_wire.outputs[0]
                            ins = self.change_instance_port_wire(ins, _port, right_wire)
                            right_wire.outputs = [(ins, _port)]
                            assign_to_remove.append(item)
                        elif node_name in self.output_nodes:
                            wire = self.wires[item.right.var.name]
                            wire.add_port(item.left.var, None)
                            self.output_to_assign[item.left.var] = item
                        # 如果node_name是在extra_ports里,则新增一个output节点
                        else:
                            print((list(self.output_nodes.keys())))
                            break
                    elif item.right.var.name in self.input_nodes:
                        node = self.input_nodes[item.right.var.name]
                        wire = self.wires[item.left.var.name]
                        wire.input = [node.ast_node]
                    elif item.left.var.name in self.output_nodes:
                        # 如果左值在output节点里，则视为wire输出到output中
                        node = self.output_nodes[item.left.var.name]
                        wire = self.wires[item.right.var.name]
                        wire.outputs.append((node.ast_node, None))
                        self.output_to_assign[item.left.var.name] = item
                elif isinstance(item.right.var, Pointer):
                    node_name = item.right.var.var.name + "_" + str(item.right.var.ptr.value)
                    # 如果node_name是在extra_ports里,则新增一个input节点
                    if node_name in self.extra_ports:
                        self.add_input_decl(node_name)
                        self.extra_ports.remove(node_name)
                        item.right.var = Identifier(name=node_name, lineno=item.right.var.lineno)
                    elif node_name in self.wires:
                        right_wire = self.wires[node_name]
                        left_var_name = item.left.var.name
                        if left_var_name in self.wires:
                            left_wire = self.wires[left_var_name]
                            ins, _port = right_wire.input[0]
                            self.change_instance_port_wire(ins, _port, left_wire)
                            left_wire.input = ((ins, _port),)
                            assign_to_remove.append(item)
                            assign_to_remove.append(right_wire.wire)
                            self.wires.pop(node_name)
                        else:
                            item.right.var = Identifier(name=node_name, lineno=item.right.var.lineno)
                    elif node_name in self.input_nodes:
                        left_name = item.left.var.name
                        if left_name in self.wires:
                            wire = self.wires[left_name]
                            wire.input = [item.right.var]

        items = [x for x in items if x not in assign_to_remove]
        items = [x for x in items if (isinstance(x, Decl) and x.list[0] not in assign_to_remove) or not isinstance(x, Decl)]
        return items

    def parse_instance(self, items):
        for item in items:
            if isinstance(item, InstanceList):
                instance = item.instances[0]
                self.instances[instance.name] = instance
                node = Node(name=instance.module+instance.name, ast_node=instance)
                self.nodes[instance.name] = node
                for port in instance.portlist:
                    if isinstance(port.argname, Pointer):
                        port_name = port.argname.var.name + '_' + str(port.argname.ptr.value)
                        if port_name in self.wires:
                            port.argname = Identifier(name=port_name, lineno=port.argname.lineno)
                            self.wires[port.argname.name].add_port(instance, port)
                        elif port_name in self.input_nodes:
                            self.input_nodes[port_name].add_child(instance, port)
                    elif isinstance(port.argname, Identifier):
                        if port.argname.name in self.wires:
                            self.wires[port.argname.name].add_port(instance, port)
                        elif port.argname.name in self.input_nodes:
                            self.input_nodes[port.argname.name].add_child(instance, port)
                        else:
                            pass

                self.nodes[instance.name] = Node(name=instance.module + instance.name, ast_node=instance)

    def parse_ast(self):
        self.module.items = self.parse_decl(self.module.items)
        self.parse_instance(self.module.items)
        self.module.items = self.parse_assign(self.module.items)


def main():
    inputFile = args.src_file
    outputFile = args.result_file

    ast, directives = parse([inputFile])
    nt = NodeTree(ast)

    nt.order()
    codegen = ASTCodeGenerator()
    rslt = codegen.visit(nt.ast)
    with open(outputFile, "w") as f:
        f.write(rslt)


if __name__ == '__main__':
    main()
