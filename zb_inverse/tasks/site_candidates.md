# Field sites for the zb inverse test: candidates (search of 14 Nov 2026)

What the test needs at one plot: near-surface SWC (0-0.30 m, hourly), a well within about 50 m (WT mostly 0.2-1.5 m, moving at least 0.2 m), hourly P, and ET (LE, or full meteorology), on mineral, non-inundated, undrained, non-irrigated, non-tidal, unfrozen soil, with at least one warm season of overlap.

Tags: **[V]** = verified in the data, a variable list or a metadata table (source given). **[I]** = inferred from a description or a paper.
Small samples and metadata caches are in `/storage/project/r-rbras6-0/jjason3/pinn/zb_inverse/output/site_search/<source>/`. They were read as text only.

## Bottom line

- **Group A (open now): no site has all four variables (SWC, WT, P, ET) in one open archive.** Two archives have a verified co-located near-surface sensor and well at hourly or finer resolution: Point Reyes (ESS-DIVE) and Shale Hills RTH3 (HydroShare). Both lack downloadable on-site precipitation. Shale Hills has partial meteorology. Both would need rain from another source.
- **Group B: the strongest site is ICOS FR-Bil (Bilos, Landes).** It has sandy podzol, a shallow water table, and per-plot WTD and SWC profiles, with LE and P from the same station [V for the variables, I for the WT range]. ICOS needs no account, but each download requires a licence click-through, which I did not do.
- **AmeriFlux has nothing new on mineral soil.** In the BASE measurement-height table, the only sites with both SWC and WTD that are not already in your used or ruled-out lists are BR-Xpw, CA-KLP, CA-WP1, CA-WRF, US-ABe and US-UTP [V]. Of these, only BR-Xpw (a seasonally flooded Pantanal pasture) is not peat, organic muck or saline marsh.
- **Access changes in 2026 move several of your suggested sources to group B.** NEON downloads need a login or token since 30 Jun 2026 (anonymous call returned HTTP 403 "Access Denied" [V]). The EDI/LTER REST API refuses anonymous calls since 30 Jul 2026 (HTTP 403 "Public Access not authorized" [V]). This covers North Inlet, Kellogg, Coweeta, Hubbard Brook and Konza.

---

## Group A: downloadable now with no account or personal details

### A1. Point Reyes field site, coastal grassland hillslope (LBNL / ESS-DIVE)
- **Location:** 37.971 N, 122.730 W, Marin County, CA [V: package metadata].
- **Cover and soil:** annual coastal grassland on a small hillslope [V: abstract]. Mineral soil; texture tables are in the companion package `ess-dive-5b6529c66312157-20240718T120020410` (values not read) [I].
- **Soil moisture:** Teros 12 at 10, 30, 60 and 90 cm, at 5 loggers SM1-SM5 along the slope, 15-min [V: abstract and data dictionary `SMx_z6_dd.csv`]. The assignment of depths to ports 1-4 was not checked.
- **Water table:** pressure transducers in about 4 m wells at 5 locations (2020-2024), plus 1 m shallow wells (2023-2024). Hourly depth below ground in column `WLdepthMeter`; -9999 means WT below the sensor [V: `WL_dd.csv`, `SHWL_dd.csv`].
- **Co-location:** SM2-STOP, SM3-S-MID and SM5-SBOT are each about 1 m apart [V: computed from `SM_locations.csv` and `GWT_locations.csv`]. SM1 is about 15 m from well M-SS.
- **Observed WT range:** at STOP, Jan-Mar 2021 0.72-1.16 m, Nov-Dec 2021 0.49-1.16 m. Apr-Jul 2021 falls 1.1 to 3.5 m [V: first 400 kB of `WL_top2024B.csv` streamed through awk].
- **Precipitation and ET:** not in the package [V]. The companion package says flux data are available from the author [V]. No AmeriFlux tower lies within about 10 km [V: AmeriFlux site list], and no USGS rain gauge in NWIS nearby [V].
- **Period:** Nov 2020 to Nov 2024, ongoing [V].
- **URLs:**
  - Package: https://data.ess-dive.lbl.gov/view/ess-dive-9e9407a156e60ed-20241223T090100071
  - API: https://api.ess-dive.lbl.gov/packages/ess-dive-9e9407a156e60ed-20241223T090100071
  - `WL_top2024B.csv`: https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-f9e6620d4d07676-20241218T081040270
  - `SM2_z6_11607_12Nov24.csv`: https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-0e2c99889b7f00a-20241218T081040239
  - Locations: `ess-dive-08ac491a9e589c0-20241218T081203863` (SM) and `ess-dive-c26944b4f1ed377-20241218T081203850` (wells).
- **Licence:** CC-BY-4.0 [V].
- **Main risks:**
  - No on-site P or ET; rain must come from a nearby station (none verified).
  - The WT is in the 0.2-1.5 m band only in the cool wet season (Nov-Mar), not the warm season. Winters are mild, so this is acceptable only if the warm-season rule exists mainly to exclude frozen soil.
  - The hillslope implies lateral inflow.
  - The deep wells use about 4 m sensors, so -9999 marks summer.

### A2. Susquehanna Shale Hills CZO, valley-floor triangle "RTH3" (Penn State / HydroShare)
- **Location:** 40.6645 N, 77.9055 W, central PA [V: ReadMe].
- **Cover and soil:** deciduous forest. Colluvial silt loam with shale fragments (Ernest/Blairton series) [I: CZO literature]. Mineral.
- **Soil moisture:** Decagon ECH2O at 0.1, 0.3 and 0.5 m, in three nests, 10-min, 2009 to mid-2012 [V: ReadMe and file header]. Sensors 1, 4 and 7 are the 0.1 m sensors.
- **Water table:** Wells 1, 2 and 3 are 1.10, 1.96 and 2.31 m deep. Druck transducers record depth to water every 10 min, 2009-2012 [V: ReadMe, file header]. Values cap at the well depth when the well is dry [V: data].
- **Co-location:** each sensor nest is at exactly the well coordinates and ground elevation (W1/nest 1-3, W2/nest 4-6, W3/nest 7-9) [V: ReadMe of both resources].
- **Observed WT range, 2010:**
  - Well 2: Jan-Apr 0.37-0.85 m, May 0.51-1.06 m, Jun 0.74-1.45 m, Jul 1.45-1.81 m, Aug dry (>1.96 m), Sep 1.36-1.96 m.
  - Well 1: dry (>1.08 m) Jun-Aug; 0.47 m after a September storm.
  - Well 3 is mostly flagged.
  - [V: whole 2010 file, 2.5 MB, in `output/site_search/hydroshare/shalehills/gw2010.txt`]
- **Precipitation:** the QC hourly precipitation entry (2006-2013) on HydroShare is only a link to the defunct czo.psu.edu, which returns 404 [V]. The CZO HTML pages are in the Wayback Machine, but no data-file links were found in them [V].
- **ET:**
  - The flux tower about 190 m away has 10-min net radiation, T, RH, pressure and PAR for 2009-2013 as real files [V: header]. There is no wind and no LE in those files.
  - The LE resource (2009-2016) is a dead link [V].
  - AmeriFlux US-SSH (same tower) has P and LE, but BASE covers only 2016-2021 [V: AmeriFlux availability API and your local file], so it does not overlap 2009-2012.
- **URLs:**
  - Groundwater: https://www.hydroshare.org/resource/0403737ef5b147c3916ab30d565db79c/ (file `2010_CZO_RTH3_SH_ten_min_Groundwater.dat`; also 2009, 2011, 2012).
  - Soil moisture: https://www.hydroshare.org/resource/fa5e043282894e7e97a663a99f9506ad/ (`20xx_CZO_RTH3_SH_ten_min_Soilmoisture.dat`).
  - Tower meteorology: https://www.hydroshare.org/resource/4c3da3f64b2f436cae68f45f77f4e3be/
- **Licence:** CC BY 4.0 [V: HydroShare rights].
- **Main risks:**
  - No public P or LE for 2009-2012. P must come from a PI request or an off-site gauge such as the State College airport (not checked).
  - Valley floor next to the first-order stream: convergent lateral inflow.
  - Wells go dry in August.
  - Rock fragments.
  - Usable windows are Mar-Jun and Sep-Nov.

### A3. Illinois Climate Network, Bondville station (Illinois State Water Survey, WARM)
- **Location:** Champaign County, IL [V]. Sod plot at the Bondville research site [I].
- **Soil:** silt loam (Drummer/Flanagan) [I].
- **Water table:** well #1, "Bondville (ICN)", 1996-present. Daily depth series 2001-2026 is embedded in the station page and exported as CSV [V].
  - Values in 2018-2020: spring 0.35-3.5, summer 2.5-6.0, autumn 2020 up to 8.8 [V].
  - Units are not stated. Feet are assumed from the station metadata, i.e. about 0.1-1.1 m in spring and 0.8-1.8 m in summer [I].
  - The WARM chart wells are monthly visits; this one is the ICN well [V: page title "Bondville (ICN)"].
- **Soil moisture:** the ICN reports soil moisture in water fraction by volume [V]. ICN measurements are taken every 10 s to hourly and averaged for the hour and the day [V: ICN page]. The public archive page offers daily and monthly files only [V]. Depths and hourly availability are not verified.
- **Co-location:** "Shallow groundwater wells are co-located at the stations" [V: ICN page]. Distance not given.
- **Precipitation and ET:** ICN hourly weather (precipitation, radiation, T, RH, wind) [I: ICN description; not verified as hourly downloads].
- **URLs:**
  - Station page: https://warm.isws.illinois.edu/warm/groundwater/station.asp?well=1
  - ICN description: https://warm.isws.illinois.edu/warm/climnet/abouticn.asp
  - Daily data: https://warm.isws.illinois.edu/warm/datatype.asp
- **Licence:** not stated; "Data Use" page not read. A registration banner exists in the page source but is commented out [V].
- **Main risks:** WT is daily only; SWC may be daily only; depth units not stated; tile-drained landscape around the plot.

### A4. Kopčany poplar SRC stands, western Slovakia (TU Dresden / PANGAEA)
- **Location:** three stands, S5-N, S4-D and S2-F, at about 48.45-48.47 N, 17.37-17.63 E [V: PANGAEA events].
- **Cover and soil:** hybrid poplar short-rotation coppice on lowland soils [V]. Texture not read.
- **Soil moisture:** SM150T at 0.3, 0.6 and 0.9 m only (no shallower sensor), 15-min until 16 Mar 2020 and coarser afterwards [V: parameters and abstract].
- **Water table:** well 3 m from each soil pit, Keller transducer [V: abstract].
- **Precipitation:** throughfall collectors, hourly [V].
- **Meteorology:** separate 30-min dataset [V: title]. Parameters not read.
- **Observed WT range:** not obtained. The sites are described only as "higher", "low" and "fluctuating" groundwater [V].
- **URLs:**
  - Soil and groundwater: https://doi.pangaea.de/10.1594/PANGAEA.951066 (add `?format=textfile`)
  - Meteorology: https://doi.pangaea.de/10.1594/PANGAEA.951254
- **Licence:** CC-BY-4.0 [V].
- **Main risks:** shallowest sensor at 0.3 m, at the edge of 0-0.30 m; no LE; strong poplar root uptake; WT range unknown, possibly river-influenced.

---

## Group B: needs an account, a licence acceptance or a PI request

### B1. FR-Bil, Bilos, Landes forest, France (ICOS; no account, but a licence click-through on every download)
- **Location:** 44.494 N, 0.956 W [I: ICOS station page via search].
- **Cover and soil:** maritime pine plantation (replanted). Haplic podzol, loamy sand to sandy loam, with a patchy iron hardpan at about 75 cm [I: ICOS station page].
- **Variables:**
  - ETC L2 Meteo: SWC_1-8, WTD, P, TA, RH, SW_IN, WS [V: ICOS metadata column list via SPARQL].
  - ETC L2 Meteosens (per sensor): SWC_1_1_1-SWC_1_7_1 and SWC_2_1_1-SWC_2_7_1 profiles, SWC_4_1_1, SWC_5_1_1, WTD_1_1_1-WTD_3_1_1 [V].
  - ETC L2 Fluxnet HH v1.3_r1 (2019-2026): LE_F_MDS, P_F, TA_F, VPD_F, WS_F [V].
  - Half-hourly [V: product type]. Sensor depths are in the BADM/VARINFO inside the L2 ARCHIVE; not read.
- **Co-location:** WTD_1 and WTD_2 share the horizontal index of SWC profiles 1 and 2. In FLUXNET naming this marks the same soil plot [I].
- **Period:** 2019 to Aug 2026 [V].
- **WT range:** wet Landes regionally 0-0.5 m in winter and about 1-1.5 m in summer. A continuous piezometer exists at the Bilos plot (Deirmendjian et al. 2018, Biogeosciences) [I].
- **URLs:**
  - L2 Meteo: https://meta.icos-cp.eu/objects/8puLUPpif6GiiVRSQbhVnRsA
  - L2 Meteosens: https://meta.icos-cp.eu/objects/z3RB-bxCZ56yThkccIDCqxuM
  - Fluxnet HH: https://meta.icos-cp.eu/objects/xFrwVCA1RxiQ4UyGinPgeqSZ
  - Station: https://meta.icos-cp.eu/resources/stations/ES_FR-Bil
- **Licence:** CC BY 4.0, accepted at download [I: ICOS FAQ].
- **Main risks:** drainage ditches ("crastes") are common in the Landes and may control the WT; the patchy hardpan at 75 cm; plantation management (clear-cut and replanting).
- **Why first:** closest physical analogue to US-NC2 and US-HB2 (sand, shallow WT, pine).

### B2. BE-Bra, Brasschaat (De Inslag), Belgium (ICOS licence click-through)
- **Cover and soil:** Scots pine. Albic Hypoluvic Arenosol (aeolian cover sand 1.2-2.25 m thick) over Campine clay. Organic layer 6 cm [I: ICOS station page and DEIMS].
- **Variables:**
  - L2 Meteo: SWC_1-5, WTD, P, TA, RH, SW_IN, WS, G [V].
  - Meteosens: SWC profiles 1-4 (5 depths each) and WTD_1_1_1-WTD_4_1_1, so H indices match [V].
  - Fluxnet HH 2020-2026: LE_F_MDS, P_F, TA_F, VPD_F, WS_F [V].
- **Co-location:** inferred from the matching H indices [I].
- **Period:** 2020 to Aug 2026; final L2 for 2020-2023 [V].
- **WT range:** perched WT at 0.8-2.5 m, usually 1.2-1.5 m [I: DEIMS and the ANAFORE paper].
- **URLs:**
  - L2 Meteo: https://meta.icos-cp.eu/objects/0qaWXr70FX3vP3N_uZhhBPY7
  - Meteosens L2: https://meta.icos-cp.eu/objects/YmJMz6xSnFqhn-vVA0UKL69H
  - Fluxnet HH: https://meta.icos-cp.eu/objects/8DiB32-pJL4rY2j9M9IdP8HN
- **Licence:** CC BY 4.0 [I].
- **Main risk:** the WT is often at or below 1.2-1.5 m in sand, so a 0.05-0.1 m sensor may barely respond (the US-Ho1 failure mode); perched on clay; possible forest ditches.

### B3. SE-Svb, Svartberget, Krycklan, Sweden (ICOS licence click-through)
- **Cover and soil:** Norway spruce / Scots pine on stony till, shallow gneiss bedrock [I: ICOS labelling report].
- **Variables:**
  - L2 Meteo: SWC_1-5, WTD, P [V].
  - Meteosens: SWC profiles 1-4 and WTD_1_1_1, WTD_2_1_1 [V].
  - Fluxnet HH 2019-2026: LE_F_MDS, P_F, WS_F [V].
  - Older ICOS Sweden station page lists SWC at 0-0.06 m (vertical), 0.05, 0.1, 0.3 and 0.5 m [I].
- **WT range:** below 2 m in most Krycklan wells and below 1 m on mesic-moist soils [I: Erdbrügger et al. 2023 ESSD; HESS 2022].
- **URLs:**
  - L2 Meteo: https://meta.icos-cp.eu/objects/rRVNSjaB259uB34FTgCapxZL
  - Meteosens: https://meta.icos-cp.eu/objects/LJb4al4DIlP4ubJyB-W3K1AM
  - Fluxnet HH: https://meta.icos-cp.eu/objects/J8fsLFJ-v5264yF6bZHDuY4y
- **Main risks:** mor humus layer (organic O horizon) over the till; stony; lateral flow on slopes; snow and frozen soil Nov-Apr, so only Jun-Sep is usable.

### B4. BE-Lcr, Lochristi poplar SRC, Belgium (ICOS licence click-through)
- **Variables:**
  - L2 Meteo 2019-2022 (final): SWC_1-5, WTD, P [V].
  - Meteosens: SWC profiles 1-5 and WTD_1_1_1-WTD_5_1_1, one well per profile by H index [V].
  - Fluxnet HH L2: LE_F_MDS, P_F [V].
- **Soil and WT:** not checked. Former farmland, sandy loam, shallow WT [I].
- **URLs:**
  - Meteo: https://meta.icos-cp.eu/objects/nrKlUQmgHacKZ3YlXZjDJ59H
  - Meteosens: https://meta.icos-cp.eu/objects/mCWlD_xgJ4lyUJ8L1S15lKx-
  - Fluxnet: https://meta.icos-cp.eu/objects/AGt5bc2Y6euDplFUqMn2qgi4
- **Main risks:** possible field ditches; WT range unknown; SRC harvest cycles.

### B5. BR-Xpw, XomanoFlux, northern Pantanal, Brazil (AmeriFlux account)
- **Location and cover:** -16.373, -56.461. Grazed "hyperseasonal" pasture, flooded Nov-Apr, dry May-Oct [V: site page].
- **Variables in BASE:** WTD_PI_1 (height 0), WTD_PI_2, SWC_1_1_1-SWC_1_5_1 at 0.05, 0.1, 0.3, 0.6 and 1.0 m, LE, P [V: BASE_MeasurementHeight_20260908.csv].
- **Period:** 2017-2022 [V: availability API].
- **Soil:** not reported on the site page [V].
- **URL:** https://ameriflux.lbl.gov/sites/siteinfo/BR-Xpw
- **Licence:** CC-BY-4.0 [V].
- **Main risks:**
  - Inundated half the year.
  - The dry season has almost no rain, so rises come from flooding rather than rain.
  - The WTD sign convention for PI variables must be checked (the CL-SDF lesson).
  - Probable perched or plinthic layers.

### B6. NEON LENO + TOMB and DELA + BLWA, Alabama floodplain forests (NEON account or token)
- **Variables:**
  - Terrestrial towers LENO and DELA: soil water content profiles (DP1.00094.001), EC (DP4.00200.001), precipitation. Soil water content from Jun 2017 [V: NEON product and site lists].
  - Co-sited river stations TOMB and BLWA: groundwater wells (DP1.20100.001), from Jul 2018 [V].
- **Distances:** TOMB well 001 is 76 m from the LENO tower; BLWA well 002 is 59 m from the DELA tower [V: computed from NEON location API]. Soil-plot coordinates are not exposed in the location API, so the plot-to-well distance is unknown.
- **WT range:** not checked.
- **URLs:**
  - Well location: https://data.neonscience.org/api/v0/locations/TOMB.AOS.groundwater.well.001
  - Groundwater product: https://data.neonscience.org/data-products/DP1.20100.001
- **Licence:** CC BY since 30 Jun 2026 [I: NEON blog].
- **Main risks:** WT controlled by river stage (lock-and-dam pools); overbank flooding; clayey alluvium; well probably more than 50 m from the soil plots.

### B7. Yahara River watershed hydrologic sites, Wisconsin (EDI login, package edi.128.1)
- **Variables:** soil moisture, soil temperature, matric potential and depth to groundwater under forest, grassland, wetland, urban turf and corn, 2011-2016 [V: abstract via the DataONE index].
- **Not obtained:** depths, time step, distances and WT range (EDI refused the anonymous metadata call [V]).
- **URL:** https://portal.edirepository.org/nis/mapbrowse?scope=edi&identifier=128
- **Main risks:** uplands likely have deep WT and the wetland may be organic [I]; corn may be tile-drained [I].

---

## The three sites I would try first
1. **FR-Bil (ICOS).** It is the only candidate with LE, P, a per-plot well and a near-surface SWC profile at one station, on sand with a shallow WT. That is the setting where US-ONA, US-PAS, US-NC2 and US-HB2 worked. Clicking the ICOS licence takes one step. Check the crastes and the hardpan first, using the WTD_n series and the BADM.
2. **Point Reyes (ESS-DIVE).** It is open now, with three verified well and sensor pairs 1 m apart, hourly WT in the 0.5-1.2 m band with storm rises, and a 10 cm sensor. Winter ET is small, so q_surf is close to -P. It does need a rain gauge.
3. **Shale Hills RTH3 (HydroShare).** It is open now, with an exact sensor and well co-location, 10-min data, silt loam with the WT at 0.4-1.5 m in spring and early summer, and on-site Rn/T/RH. It is usable only once 2009-2012 precipitation is found: ask the PI (C. Duffy) for the CZO archive, or use an off-site gauge.

BE-Bra is next if a fourth site is wanted, but its WT depth is the main concern.

## Looked at and did not qualify (one line each)
- **AmeriFlux BASE overall:** scanned every site with WTD in the measurement-height table [V]. New WTD+SWC sites besides BR-Xpw: CA-KLP (bog), CA-WP1 (fen), CA-WRF (fen, 2024+), US-ABe (Samsula muck, seasonal wetland [V: site page]), US-UTP (Phragmites at the Great Salt Lake margin).
- **US-MAC:** no BASE published (empty publish_years [V]). FLUXNET-CH4 has WTD but no SWC variable [V: fluxnet.org variable table]. PI request only.
- **US-DPW:** BASE 2013-2017 has neither WTD nor SWC [V].
- **US-Tw3:** SWC at 0.1 and 0.2 m, no WTD [V]; peat island, irrigated alfalfa.
- **US-Bi1 / US-Bi2:** SWC at 0.15 m, no WTD [V]; Delta peat, irrigated.
- **US-SP1 / US-SP2 / US-SP3:** SP1 has SWC at 0.15 m only; SP2 and SP3 have neither [V].
- **US-NC1:** SWC at 0.2 m, no WTD [V]; ditched plantation.
- **US-ORv, US-KS1, US-KS2:** neither SWC nor WTD in BASE [V]. KS2 WT records exist only in papers (PI request).
- **US-xDS / NEON DSNY:** SWC profile 0.05-1.95 m but no WTD [V]. NEON terrestrial sites have no wells [V].
- **US-HB3:** SWC at 0.05-1.0 m but no WTD in BASE [V].
- **US-HB1 / US-HB4:** WTD without SWC [V]; tidal marsh and brackish impoundment.
- **Rest of the local "LIKELY" list** (US-MBP, BZB, BZF, EML, Skr, LA1-3, KS3-4, Atq, Brw, ICh, ICs, PHM, PLo): no WTD in BASE [V], or peat, tundra or tidal.
- **USGS NWIS:** 265 sites with soil-moisture series were scanned for wells at the same ID or within about 60 m [V: api.waterdata.usgs.gov and waterservices]. Matches that failed:
  - Gary IN BFPK-1 (dune sand): WT 2.6-3.4 m in 2016-2021 [V].
  - Uinta Basin UT U(D-1-1)19aad-1: WT 2.7-4.8 m [V].
  - Delaware County PA DE 723: 300 ft bedrock well [V].
  - Bemidji MN: deep WT at the oil-spill research site [V: well depths].
  - Mower County MN transects: tile-drained, with SWC only at 0.4 and 0.8 m [V].
  - Detroit Recovery Park: SWC only Aug 2017-Mar 2018 [V]; urban fill.
  - Holly Tree Lane, Fond du Lac WI: WT 0.3-4 m, daily only, and soil-moisture series not served [V].
  - Missouri DNR wells (MO005): SWC only since Jul 2026 [V].
  - Dixie Valley NV playa (DVPL2): WT 0.15-0.57 m but saline playa, not rain-driven [V]. DVDV: 4.6-5.4 m [V].
  - Ranch Creek meadow CO: piezometer has field readings only, and no precipitation series [V].
  - Neversink NY: in-stream piezometers with no ground reference [V].
  - Great Dismal Swamp wells: peat.
- **NEON in general:** soil moisture only at terrestrial sites, groundwater only at aquatic sites [V: product site lists]; downloads now need a token [V].
- **EDI/LTER** (North Inlet, Hubbard Brook, Coweeta, Kellogg, Konza): API needs login since 30 Jul 2026 [V: HTTP 403]. Not assessed dataset by dataset; any hit would be group B.
- **Kellogg Experimental Forest EAB riparian plots (HydroShare):** no precipitation, logger level not referenced to the ground, black-ash riparian zone [V: file headers].
- **Gensburg-Markham Prairie (HydroShare):** piezometers only; the soil-moisture data are not in the resource [V].
- **Shale Hills 2013-2019 wells and 2015-2018 valley-floor soil moisture, and Lin's RTHnet water table at 10 sites:** HydroShare entries only link to the defunct czo.psu.edu [V].
- **REC Western Australia (HydroShare):** neutron-probe soil moisture, mostly manual depth to WT; Gnangara/Gingin WT likely deep [I].
- **CERP C-111 Spreader Canal (SFWMD):** canal-controlled Everglades marl and peat [I: title and abstract].
- **Wax Lake Delta (ESS-DIVE):** deltaic, tidal or inundated [I].
- **Krycklan shallow-groundwater dataset (ESSD 2023):** groundwater only; no co-located SWC in that release [I].
- **ICOS peatlands with WTD** (CZ-wet, DE-GsB, FI-Lom, FI-Sii, FR-LGt, GL-ZaF, UK-AMo, SE-Deg, SE-Myc, SE-Sto, Abisko): peat or organic, so excluded.
- **ICOS croplands with WTD** (DE-RuS, BE-Lon, BE-Dor, FR-Lam, IT-BCi): deep WT or irrigation [I].
- **ICOS with WTD and SWC, WT range not checked:** SE-Nor, SE-Htm, DK-Sor, FR-Fon, FR-Hes, IT-SR2, IT-Cp2, NL-Loo, NO-Hur, CZ-Lnz, DK-Vng, IT-Lsn [V: columns]. These are the next leads if FR-Bil and BE-Bra fail.
- **KNMI Cabauw:** peat and clay with ditch-controlled levels; excluded by the criteria (not checked in data).
- **TERENO** (Wüstebach, Rollesbroich, Selhausen): not checked in data. Wüstebach riparian soils are organic and Selhausen's WT is deep [I].
- **Zenodo keyword searches:** only peatland, lysimeter and isotope datasets turned up; nothing usable.

---

## Checks made on this report, and the ICOS sensor layout (14 Nov, 01:19)

Done from the cached samples and from open ICOS metadata only (landing pages of `meta.icos-cp.eu` as JSON, and its SPARQL endpoint; cached in `output/site_search/icos/meta/`). No ICOS data file was requested, no licence accepted, nothing fitted.

What holds as written:
- ICOS column lists of FR-Bil and BE-Bra (L2 Meteo, Meteosens, Fluxnet HH), and the periods.
- The three FR-Bil objects resolve (19, 37 and 62 MB). Licence of each: "ICOS CCBY4 Data Licence" (CC BY 4.0), https://data.icos-cp.eu/licence.
- ESS-DIVE Point Reyes: CC BY 4.0, 26 files.
- Shale Hills 2010, recomputed from the cached file (flag A only): well 2 as stated (Jan 0.37 - 0.85 m, Jun 0.74 - 1.45, Jul 1.45 - 1.81, Aug at the well bottom).

Corrections and additions:
- BE-Bra: the Meteosens link given above is the final L2 for 2020 - 2023. The interim file to Aug 2026 is https://meta.icos-cp.eu/objects/aV0oPGAZvF9B5s50bumZt4f5.
- Shale Hills 2010: well 1 sits at its bottom (1.08 - 1.10 m) in February and June - August; no flag-A value for any well in November; well 3 has none all year.
- **FR-Bil co-location is verified, not inferred.** The ICOS metadata lists every sensor with its position and depth:

| plot | well (depth of the pressure sensor) | soil-moisture profile, sensor depths (m) | well to profile |
|---|---|---|---|
| 1 | WTD_1_1_1 (1.95 m) | SWC_1_x_1: 0.05, 0.08, 0.12, 0.24, 0.47, 1.05, 1.97 | 1.8 m |
| 2 | WTD_2_1_1 (1.70 m; 1.85 m since 13 Mar 2026) | SWC_2_x_1: 0.05, 0.08, 0.10, 0.16, 0.54, 0.94, 1.25 | 2.5 m |
| rain gauge | WTD_3_1_1 (3.46 m; 2.14 m since 16 May 2023) | SWC_4_1_1 (0.05 m only) | 21 m |

  - The two plots are about 90 m apart: two columns at one station. Rain gauges (P_1_1_1, P_2_1_1) are 50 - 90 m from them.
  - The four upper sensors of plot 1 were replaced on 5 Jul 2023 (same depths, 0.6 m away).
  - Not checked, because it needs the data: the water-table range itself, the ditches, the hardpan.

The same layout at the other ICOS stations with a water-table column on mineral soil (current deployments). "Well" is the depth of the pressure sensor: the deepest water table the well can record, not the water-table depth.

| station | cover and setting (station text) | wells, sensor depth (m) | nearest soil-moisture sensor in the top 0.30 m | reading |
|---|---|---|---|---|
| FR-Bil | pine plantation, sandy podzol, hardpan at 0.75 m; flat Landes; 960 mm/yr | 1.95, 1.85, 2.14 | 0.05 m, 1.8 and 2.5 m away | first |
| IT-SR2 | stone pine, 8 km west of Pisa, 4 m a.s.l., dry summer; 950 mm/yr | 2.62, 2.79 | 0.05 m at 3.8 m; 0.15 m at 2.9 m | second: coastal plain, no frost; soil not in the text |
| SE-Nor | spruce and pine, flat, sandy-loamy till; 586 mm/yr | 2.73, 2.07 | 0.05 m at 2.7 and 1.9 m | frost-free months only |
| DK-Sor | beech, Zealand; 640 mm/yr | 3.44, 2.64, 2.44, 2.78 | 0.05 m at 1.9 - 3.0 m | soil not in the text |
| BE-Dor | grazed permanent grassland, Condroz; 828 mm/yr | 1.40 | 0.05 m at 4.4 m | slope and soil not checked |
| FR-Fon | oak, gleyic luvisol; 679 mm/yr | 0.85 - 0.96 (five wells) | 0.05 m at 1.0 - 10.5 m (one well 40 m) | shallow wells: a perched winter water table [I] |
| FR-Hes | beech, luvisol to stagnic luvisol; 889 mm/yr | 0.80 (four wells) | 0.05 m at 0.7 - 1.9 m | as FR-Fon [I] |
| CZ-Lnz | floodplain forest at the Thaya - Morava confluence, flat; 518 mm/yr | 2.75, 3.59, 3.01 | 0.05 m at 1.9 - 3.0 m | river-controlled water table likely [I] |
| BE-Bra | pine on sand | 3.94 - 4.01 (four wells) | 0.05 m at 1.6 - 9.0 m | as B2 above |
| BE-Lon, FR-Lam, IT-BCi | croplands | 2.26; 3.21; 2.80 | 0.05 m at 1.8 - 3.5 m | tillage, crop cycles; later |
| NO-Hur, SE-Svb | boreal forest | 0.75, 1.52; 2.83, 1.33 | 0.05 m at 0 m; 0.02 m at 2.3 - 4.0 m | frost, till |
| IT-Cp2, DE-RuS, DK-Vng, NL-Loo | - | 3.75 - 5.5; 6.0; 8.68; 8.88, 9.59 | - | wells this deep: water table probably below 1.5 m |
| BE-Lcr, IT-Lsn | - | no current deployment listed | - | closed records, not looked at further |

Reading: ICOS stations follow one layout, a well 1 - 4 m from a soil-moisture profile with a 0.05 m sensor, with a rain gauge and eddy covariance at the same station. All objects carry the same ICOS licence, so one decision on that licence covers FR-Bil and the next stations (IT-SR2, SE-Nor, DK-Sor, BE-Dor).
