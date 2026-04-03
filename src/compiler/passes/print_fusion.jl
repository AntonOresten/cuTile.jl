# Print Fusion Rewrite Rule
#
# Fuses format_string intrinsic calls into print_tko calls to support
# Julia's string interpolation in kernel print statements.
#
# Julia lowers `print("hello $x")` to `print(string("hello ", x))`.
# The overlay routes `string(xs...)` → `Intrinsics.format_string(xs...)`
# and `print(xs...)` → `Intrinsics.print_tko(xs...)`.
#
# The rewrite inlines the format_string args into the print_tko call:
#
#   %1 = call format_string("hello ", %x)
#   %2 = call print_tko(%1, "\n")
#
# becomes:
#
#   %2 = call print_tko("hello ", %x, "\n")
#
# Dead format_string calls are removed by subsequent DCE.

const PRINT_FUSION_RULES = RewriteRule[
    @rewrite Intrinsics.print_tko(Intrinsics.format_string(~parts...), ~rest...) =>
             Intrinsics.print_tko(~parts..., ~rest...)
]
