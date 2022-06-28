import dataclasses
import math
from typing import List, Sequence, Set

from .lexitem import LexItem


@dataclasses.dataclass(frozen=True)
class Derivation:
    item: LexItem
    score: float = 0
    backptrs: Sequence["Derivation"] = ()

    def __bool__(self):
        return True

    @property
    def lexitems(self):
        items: Set[LexItem] = set()
        queue = [self]
        while queue:
            e = queue.pop(-1)
            queue += list(e.backptrs)
            if e.is_leaf():
                items.add(e.item)
        return items

    def is_leaf(self):
        return len(self.backptrs) == 0

    def is_unary_branch(self):
        return len(self.backptrs) == 1

    def is_binary_branch(self):
        return len(self.backptrs) == 2

    def latex_prooftree(self, is_root: bool = True) -> str:
        codes: List[str] = []

        for backptr in self.backptrs:
            codes.append(backptr.latex_prooftree(is_root=False))

        if self.is_leaf():
            codes.append(f'AxiomC{{${self.item.to_latex_equation()}$}}')
        elif self.is_unary_branch():
            codes.append(f'UnaryInfC{{${self.item.to_latex_equation()}$}}')
        elif self.is_binary_branch():
            codes.append(f'BinaryInfC{{${self.item.to_latex_equation()}$}}')
        else:
            raise NotImplementedError

        if is_root:
            codes.insert(0, '\\begin{{prooftree}}')
            codes.append('\\end{{prooftree}}')
        return '\n'.join(codes)

    def visualize(self) -> str:
        if self.is_leaf():
            return f'[[{self.item}]]'

        vis_backptrs = [b.visualize() for b in self.backptrs]

        height = max(len(x.split('\n')) for x in vis_backptrs)

        padded_vis_backptrs: Sequence[Sequence[str]] = []
        for x in vis_backptrs:
            lines = x.split('\n')
            width = len(lines[0])
            lines = ([' ' * width] * (height - len(lines))) + lines
            padded_vis_backptrs.append(lines)

        merged_vis_backptrs = [' '.join(x) for x in zip(*padded_vis_backptrs)]

        vis_selfitem = f'[[{self.item}]]'

        line_width = max(
            len(vis_selfitem), len(merged_vis_backptrs[0]))

        padding_len_selfitem = (line_width - len(vis_selfitem)) / 2
        padding_len_backptrs = (line_width - len(merged_vis_backptrs[0])) / 2

        return '\n'.join([
            ' ' * math.floor(padding_len_backptrs) + '\n'.join(merged_vis_backptrs) + ' ' * math.ceil(padding_len_backptrs),  # noqa: E501
            '-' * line_width,
            ' ' * math.floor(padding_len_selfitem) + vis_selfitem + ' ' * math.ceil(padding_len_selfitem),  # noqa: E501
        ])
