"""Consistent CLI selection of a legacy component or an exact configured site."""

from sae_pipeline.hooks.components import ComponentSpec, site_requests


def add_site_arguments(parser):
    parser.add_argument("--layer", type=int)
    parser.add_argument("--component")
    parser.add_argument("--site", help="Exact target.sites slug, preserving its configured hook path.")


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
