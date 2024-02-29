from simim.galprops import Prop, MultiProp
from simim.galprops import line_co, line_densegas, line_fir, line_fir_yang
from simim.galprops import sfr_behroozi13

# Pre-defined line luminosities
prop_co_sled = MultiProp(prop_names=['L10','L21','L32','L43','L54','L65','L76','L87','L98','L109','L1110','L1211','L1312'],
    prop_function=line_co.kamenetzky16,
    kwargs=['sfr'],units=['Lsun'],h_dependence=[0])

prop_li_co = Prop(prop_name="LCO",
    prop_function=line_co.li16,
    kwargs=['sfr'],units='Lsun',h_dependence=0)

prop_lidz_co = Prop(prop_name="LCO",
    prop_function=line_co.lidz11,
    kwargs=['mass'],units='Lsun',h_dependence=0)

prop_pullen_co = Prop(prop_name="LCO",
    prop_function=line_co.pullen13,
    kwargs=['mass','redshift','cosmo'],units='Lsun',h_dependence=0)

prop_keating_co = Prop(prop_name="LCO",
    prop_function=line_co.keating16,
    kwargs=['mass'],units='Lsun',h_dependence=0)

prop_gao_hcn = Prop(prop_name="LHCN",
    prop_function=line_densegas.hcn_gao,
    kwargs=['sfr'],units='Lsun',h_dependence=0)

prop_breysse_hcn = Prop(prop_name="LHCN",
    prop_function=line_densegas.hcn_breysse,
    kwargs=['mass'],units='Lsun',h_dependence=0)

prop_delooze_fir = Prop(prop_name="Lfir",
    prop_function=line_fir.delooze14,
    kwargs=['sfr'],units='Lsun',h_dependence=0)

prop_uzgil_fir = Prop(prop_name="Lfir",
    prop_function=line_fir.uzgil14,
    kwargs=['sfr'],units='Lsun',h_dependence=0)

prop_spinoglio_fir = Prop(prop_name="Lfir",
    prop_function=line_fir.spinoglio12,
    kwargs=['sfr'],units='Lsun',h_dependence=0)

prop_schaerer_cii = Prop(prop_name='LCII',
    prop_function=line_fir.schaerer20,
    kwargs=['sfr'],units='Lsun',h_dependence=0)

prop_yang_cii = Prop(prop_name='LCII',
    prop_function=line_fir_yang.yang22,
    kwargs=['mass','redshift'],units='Lsun',h_dependence=0)

prop_padmanabhan_cii = Prop(prop_name='LCII',
    prop_function=line_fir.padmanabhan19,
    kwargs=['mass','redshift'],units='Lsun',h_dependence=0)

prop_zhao_nii = Prop(prop_name='LNII',
    prop_function=line_fir.zhao13,
    kwargs=['sfr'],units='Lsun',h_dependence=0)

# Pre defined SFRs
prop_behroozi_sfr = Prop(prop_name="sfr",
    prop_function=sfr_behroozi13.sfr,
    kwargs=['redshift','mass'],units='Msun/yr',h_dependence=0)

prop_behroozi_mass = Prop(prop_name="stellar mass",
    prop_function=sfr_behroozi13.stellarmass,
    kwargs=['redshift','mass'],units='Msun',h_dependence=0)


# Aliases for backwards compatibility
prop_delooze_cii = prop_delooze_fir
prop_uzgil_cii = prop_uzgil_fir
prop_spinoglio_cii = prop_spinoglio_fir