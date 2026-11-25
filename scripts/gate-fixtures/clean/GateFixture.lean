namespace GateFixture

theorem ok : 1 + 1 = 2 := rfl

def «/-» : Nat := 0
def rawStr : String := r#"not a sorry or admit"#
def interpStr : String := s!"prefix {"sorry"} suffix"
def sorry₁ : Nat := 0

end GateFixture

private theorem privateOkOutsideNs : True := trivial
