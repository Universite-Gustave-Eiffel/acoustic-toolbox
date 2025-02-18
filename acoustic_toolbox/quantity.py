"""The Quantity module provides two classes to work with quantities and units."""

from acoustic_toolbox.standards.iso_tr_25417_2007 import REFERENCE_PRESSURE

quantities = {
    "pressure": ("Pressure", "pascal", True, "p", "$p$", REFERENCE_PRESSURE),
}
"""Dictionary with quantities. Each quantity is stored as a tuple."""

units = {
    "meter": ("meter", "m", "$m$"),
    "pascal": ("pascal", "Pa", "$Pa$"),
}
"""Dictionary with units. Each unit is stored as a tuple."""


class Unit:
    """Unit of quantity.

    Note:
      Perhaps inherit from tuple or [`collections.namedtuple`][collections.namedtuple]?
    """

    def __init__(self, name: str, symbol: str, symbol_latex: str):
        """Initialize the unit.

        Args:
          name: Name of the unit.
          symbol: Symbol of the unit.
          symbol_latex: Symbol of the unit in LaTeX.
        """
        self.name = name
        """Name of the unit."""

        self.symbol = symbol
        """Symbol of the unit."""

        self.symbol_latex = symbol_latex
        """Symbol of the unit in LaTeX."""

    def __repr__(self):
        return "Unit({})".format(self.name)

    def __str__(self):
        return self.name


class Quantity:
    """Quantity."""

    def __init__(
        self,
        name: str,
        unit: Unit,
        dynamic: bool,
        symbol: str | None = None,
        symbol_latex: str | None = None,
        reference: float = 1.0,
    ):
        """Initialize the quantity.

        Args:
          name: Name of the quantity.
          unit: Unit of the quantity.
          dynamic: Dynamic quantity (`True`) or energetic (`False`).
          symbol: Symbol of the quantity.
          symbol_latex: Symbol of the quantity in LaTeX.
          reference: Reference value of the quantity.
        """
        self.name = name
        """Name of the quantity."""

        self.symbol = symbol
        """Symbol of the quantity."""

        self.symbol_latex = symbol_latex
        """Symbol of the unit in LaTeX."""

        self.unit = unit
        """Unit. See [`Unit`][acoustic_toolbox.quantity.Unit]."""

        self.dynamic = dynamic
        """Dynamic quantity (`True`) or energetic (`False`)."""

        self.reference = reference
        """Reference value of the quantity."""

    def __repr__(self):
        return "Quantity({})".format(self.name)

    def __str__(self):
        return self.name

    @property
    def energetic(self) -> bool:
        """Energetic quantity (`True`) or dynamic (`False`)."""
        return not self.dynamic


def get_quantity(name: str) -> Quantity:
    """Get quantity by name. Returns instance of [`Quantity`][acoustic_toolbox.quantity.Quantity].

    Args:
      name: Name of the quantity.

    Returns:
      Instance of [`Quantity`][acoustic_toolbox.quantity.Quantity].
    """
    try:
        q = list(quantities[name])
    except KeyError:
        raise ValueError("Unknown quantity. Quantity is not yet specified.")
    try:
        q[1] = Unit(*units[name])
    except KeyError:
        raise RuntimeError(
            "Unknown unit. Quantity has been specified but unit has not."
        )

    return Quantity(*q)
