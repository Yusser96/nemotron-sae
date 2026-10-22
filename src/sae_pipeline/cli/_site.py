"""Consistent CLI selection of a legacy component or an exact configured site."""

from sae_pipeline.hooks.components import ComponentSpec, site_requests


def add_site_arguments(parser):
    parser.add_argument("--layer", type=int)
    parser.add_argument("--component")
    parser.add_argument("--site", help="Exact target.sites slug, preserving its configured hook path.")


def sae_variant(cfg, args=None) -> tuple[str, int, int]:
    """Arch, width and L0 of the SAE variant; command-line values override the first sweep entry."""
    arch = getattr(args, "arch", None) or cfg.sae.arch
    d_sae = cfg.sae.d_sae
    width = getattr(args, "width", None) or (d_sae if isinstance(d_sae, int) else d_sae[0])
    l0_target = cfg.sae.l0_target
    l0 = getattr(args, "l0", None) or (l0_target if isinstance(l0_target, int) else l0_target[0])
    return arch, width, l0


def variant_stem(arch, width, l0) -> str:
    return f"{arch}_w{width}_l0_{l0}"


def cli_site(cfg, args, parser):
    if args.site:
        if args.layer is not None or args.component is not None:
            parser.error("Use --site or both --layer and --component")
        targets = [s for s in (cfg.target.sites or []) if s.slug == args.site]
        if not targets:
            parser.error(f"Unknown target.sites slug {args.site!r}")
    else:
        if args.layer is None or args.component is None:
            parser.error("Supply --site or both --layer and --component")
        targets = [s for s in (cfg.target.sites or [])
                   if (s.layer, s.component) == (args.layer, args.component)]
        if len(targets) > 1:
            parser.error("Several exact hooks match this component; select --site")
        if not targets:
            return ComponentSpec.parse(args.layer, args.component), None
    target = targets[0]
    spec, site = site_requests([target])[target.slug]
    spec.site_slug = target.slug
    return spec, site
