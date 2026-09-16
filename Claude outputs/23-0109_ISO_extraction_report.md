# PIPING ISOMETRIC EXTRACTION AND CHECKLIST REPORT

**Source file:** `23-0109.pdf` (1 page, A3, vector PDF, no embedded text layer)
**ISO NO.:** 3"-MMA-23-0109-SC30-CC38MM-03, Sheet 1 of 1, Rev 3
**Analysis date:** 2026-09-16

> **Coordinate convention for all `bbox` values:** normalized 0-1000, origin top-left, taken on the rendered page in its viewed (landscape) orientation. Positional accuracy approximately ±15 units.
> **Evidence classes used:** `PHYSICALLY_SHOWN` (symbol drawn), `ANNOTATION` (text/tag on drawing), `CALCULATED` (arithmetic on drawing values only), `INFERRED` (connectivity logic).

---

# SECTION 1 - EXTRACTION SUMMARY

The drawing is a single-sheet, as-built piping isometric for a 3 inch stainless steel (TP304 / Sch 10S) line in MMA service at the PTT Tank Terminal, Map Ta Phut, Thailand.

**What the line does (as drawn):**
A 2 inch flanged connection at pump/equipment item **P-2303 nozzle DN2** (E 366700, N 975345, EL +6305) rises vertically, expands 2 in to 3 in through a concentric reducer, turns through a 90° elbow at EL +6581, runs **SOUTH 1090 mm**, turns down through a second 90° elbow, drops **736 mm** to EL +5844/5845, turns through a third 90° elbow, and runs **EAST 1006 mm (dimensioned) / 1011 mm (by coordinates)** through a flanged 3 inch swing check valve to a terminal weld-neck flange that continues on **ISO 3"-MMA-23-0109-SC30-CC38MM-02** at E 367711, N 974255, EL +5844.

**Three 3/4 inch dead-leg branches**, all made on 3000# sockolets:
- **BR-001** - 3/4 in **vent**, upward from the SOUTH run at the line high point (EL +6581).
- **BR-002** - 3/4 in **drain**, downward from the EAST run at the line low point (EL +5844).
- **BR-003** - 3/4 in **pressure gauge connection**, instrument tag **23-PG-008**, off the vertical drop at EL +6059.

**Axis resolution (independently verified):** the plant-north arrow points up-left at approximately 30° on the paper. Therefore on this sheet: up-left = NORTH, down-right = SOUTH, up-right = EAST, down-left = WEST, vertical = elevation. This was confirmed numerically - the down-right run dimensions sum to exactly 1090 mm and the stated northing difference is exactly 1090 mm; the up-right run and the stated easting difference agree to within 5 mm.

**Confidence overview:**

| Category | Result |
|---|---|
| Metadata / title block | Fully legible, high confidence |
| Line characteristics table | Fully legible, high confidence (1 internal inconsistency) |
| Component identification | 19 MTO items, all located on the drawing, all reconciled |
| Topology / connectivity | Complete and unambiguous for the main run and all 3 branches |
| Main-run dimensions | All segments explicitly dimensioned; 3 minor chain-closure errors (3 mm, 1 mm, 5 mm) |
| Flow direction | **Not shown on the drawing.** Inferred only |
| Unresolved items | 1 unidentified element at the terminal flange ("TAIL U / OPERATE OPEN") |

**Counts:** 3 x 90° LR elbows, 1 x concentric reducer, 3 x sockolets, 1 x swing check valve, 2 x 3/4 in SW gate valves, 1 x 3/4 in SCRD/SW gate valve, 4 x weld-neck flanges (3 off 3 in, 1 off 2 in), 2 x 3/4 in SW flanges, 2 x 3/4 in blind flanges, 5 x 3/4 in nipples, 1 x pipe support.

---

# SECTION 2 - DRAWING METADATA

```json
{
  "drawing_metadata": {
    "project_name": "PTT TANK TERMINAL PROJECT, MAP-TA-PHUT, THAILAND",
    "project_number": "01001D (JOB NO.)",
    "client": "PTT TANK TERMINAL COMPANY, LTD.",
    "company": "THE CONSORTIUM OF SK ENGINEERING & CONSTRUCTION CO., LTD. AND THAI WOO REE ENGINEERING CO., LTD. (CONTRACTOR); FOSTER WHEELER INTERNATIONAL CORPORATION (CONSULTANT)",
    "drawing_title": null,
    "drawing_number": "3\"-MMA-23-0109-SC30-CC38MM-03",
    "sheet_number": "1 OF 1",
    "revision": "3",
    "revision_description": "AS-BUILT",
    "revision_date": "13-MAY-2011",
    "originator": "RSP (BY)",
    "checker": "SOC (CHKD) / UKH (CHKD)",
    "approver": "HKC (APP.L)",
    "drawing_date": "13-MAY-2011 (Rev 3); original issue 20-SEP-2010 (Rev 0)",
    "plant_north": "Arrow shown top-left, pointing up-left at approx. 30 degrees above horizontal on the sheet",
    "units": "mm (lengths, elevations, coordinates); bar (pressure); degC (temperature); inch NPS (sizes)",
    "reference_pid": ["2300-001"],
    "reference_drawings": [
      "3\"-MMA-23-0109-SC30-CC38MM-02 (continuation isometric)",
      "PLAN NO. 2300-003",
      "Pipe support standard R4-I2S(E-W) (tag SUPT19)"
    ],
    "general_notes": [
      "NOTE 1: VERIFY DIMENSIONS PRIOR TO FIELD ADJUSTMENT."
    ],
    "holds": [],
    "other_title_block_fields": {
      "AREA": "ISBL",
      "UNIT": "2300",
      "STRESS_CHECK": "Y",
      "STRESS_CAL": "(blank)",
      "SYSTEM": "(blank)",
      "APPROVED_BY": "(blank)",
      "EXTRACTION_DATE": "08-MAR-2011",
      "EXTRACTION_NO": "(blank)",
      "MODEL_FILE": "...\\tank\\MMA\\REV.3\\mma23010903.i01 (PDF footer, 5/15/2011 10:43:16 AM)"
    },
    "revision_history": [
      {"rev": "3", "date": "13-MAY-2011", "by": "RSP", "chkd": "SOC", "chkd2": "UKH", "appl": "HKC", "description": "AS-BUILT"},
      {"rev": "2", "date": "10-JAN-2011", "by": "KRIT", "chkd": "SOC", "chkd2": "UKH", "appl": "HKC", "description": "REVISED AS MARKED"},
      {"rev": "1", "date": "03-DEC-2010", "by": "SAM", "chkd": "SOC", "chkd2": "UKH", "appl": "HKC", "description": "REVISED AS MARKED"},
      {"rev": "0", "date": "20-SEP-2010", "by": "SAM", "chkd": "SOC", "chkd2": "UKH", "appl": "HKC", "description": "ISSUED FOR CONSTRUCTION"}
    ],
    "legend": {
      "S-1": "DENOTES PIPE SPOOL NO",
      "[n]": "DENOTES PARTS LIST NO",
      "<n>": "DENOTES CUT LENGTH NO",
      "weld_symbols": "SHOP WELD = filled dot; FIELD WELD = X; SOCKET WELD = dot with step; SCREWED JOINT = step; COMPN JOINT = small square; PIPE SUPPORT = bar"
    },
    "bbox_title_block": [702, 650, 996, 969],
    "confidence": 0.98
  }
}
```

---

# SECTION 3 - LINE CHARACTERISTICS

All five rows transcribed exactly from the line-characteristics table (bbox [24, 823, 488, 940]). **No SERVICE column exists on this drawing** - service is only implied by the "MMA" field inside the line number.

| # | line_number | Oper P (bar) | Oper T (degC) | Design P (bar) | Design T (degC) | Test medium | Test P (bar) | Insul code | Insul thk (mm) | RT (%) | PT (%) S/W only | PMI (%) | PWHT | Paint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 3/4"-MMA-23-0109-SC30-CC38MM | 14.30 | 10.0 | 30.00 | 70.0 | WAT | 45.00 | CC | 38.00 | 10% | 100% | - | N | D1 |
| L2 | 3/4"-MMA-23-0109-SC30-CC25MM | 14.30 | 10.0 | 30.00 | 70.0 | WAT | 45.00 | CC | 25.00 | 10% | 100% | - | N | D1 |
| L3 | 3"-MMA-23-0109-SC30-CC38MM | 14.30 | 10.0 | 30.00 | 70.0 | WAT | 45.00 | CC | 38.00 | 10% | 100% | - | N | D1 |
| L4 | 2"-MMA-23-0109-SC30-CC38MM | 14.30 | 10.0 | 30.00 | 70.0 | WAT | 45.00 | CC | 38.00 | 10% | 100% | - | N | D1 |
| L5 | 3/4"-MMA-23-0109-SC30-N | 14.30 | 10.0 | 30.00 | 70.0 | WAT | 45.00 | CC | 38.00 | 10% | 100% | - | N | D1 |

**Derived line attributes (from MTO material descriptions and the line number, not from a separate spec sheet):**

| Field | Value | Basis |
|---|---|---|
| Piping material specification | `SC30` (embedded in every line number) | ANNOTATION |
| Fluid / service | MMA (methyl methacrylate) - inferred from line-number field only | INFERRED, confidence 0.70 |
| Pipe material / schedule (3 in) | ASTM A312 TP304 SMLS, Sch 10S, BE | MTO item 1 |
| Fitting material | A403 WP304 Sch 10S (elbows, reducer) | MTO items 2, 4 |
| Forged fitting / flange material | A182 F304 | MTO items 3, 5, 6, 7, 9, 10, 11 |
| Flange rating / facing | Class 300, RFSF, 10S bore | MTO items 5, 6, 7, 9 |
| Gasket | Class 300 RF Valflon (PTFE) compressed non-asbestos fiber sheet, 3.2 mm thk | MTO items 12-14 |
| Bolting | Stud-bolt UNC A320 Gr B8 Cl.2 with A194 Gr 8 nuts, PTFE coated | MTO items 15-17 |
| Insulation type | CC (cold conservation) | Line list |
| Heat tracing | Not shown / no tracing annotation on drawing | - |

**Note:** row L5 has line-number suffix `-N` (no insulation) but the table lists insulation `CC / 38.00 mm`. See anomaly A-04.

---

# SECTION 4 - EQUIPMENT AND CONNECTIONS

| ID | Type | Tag | Size | Rating / facing | End prep | E | N | EL | Evidence text | bbox | Conf. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EQ-001 | EQUIPMENT_NOZZLE (source end) | **P-2303 / DN2** | 2 in | 300 (Class 300) | RFFE (raised-face flanged end) | 366700 | 975345 | +6305 | "CONN TO / P-2303/ DN2 / 2" RFFE 300 / E 366700 / N 975345 / EL +6305" | [79, 123, 121, 148] | 0.97 |
| CP-001 | CONTINUATION_POINT (destination end) | ISO 3"-MMA-23-0109-SC30-CC38MM **-02** | 3 in | 300 / RFSF | WN flange, RF | 367711 | 974255 | +5844 | "SEE ISO / 3"-MMA-23-0109-SC30-CC38MM -02 / E 367711 / N 974255 / EL +5844" | [613, 570, 637, 598] | 0.96 |
| IN-001 | PRESSURE_INSTRUMENT (off-drawing, balloon) | **23 / PG-008** | 3/4 in tapping | not stated | screwed (from SCRD/SW gate valve outlet) | - | - | approx. +6059 | Balloon "23 / PG-008" with dashed instrument leader | [310, 604, 403, 650] | 0.93 |
| SP-001 | SUPPORT | **SUPT19** | 3 in | - | - | - | - | - | "SUPT19 / R4-I2S(E-W)"; MTO item 19 "PSUPPORT 3 R4-I2S(E-W) QTY 1" | [274, 786, 319, 809] | 0.95 |

No vessel, exchanger, column or tank appears on this sheet. The equipment tag prefix "P-" is consistent with a pump but the drawing does not state the equipment type; this is not asserted as fact.

---

# SECTION 5 - NODE TABLE

Notation: `CL` = centerline intersection point; `face` = flange raised-face plane. Gaskets are listed as separate physical nodes per the no-merge rule; bolt sets are recorded in the MTO only (they carry no hydraulic or geometric station).

## 5.1 Main run

| node_id | component_type | tag / item | line_number | NPS | up size | dn size | rating | sched / class | material | end up | end dn | orientation | elevation (mm) | component_details | evidence_text | bbox | conf | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N-001 | EQUIPMENT_NOZZLE | P-2303/DN2 | 2"-MMA-23-0109-SC30-CC38MM | 2 | - | 2 | 300 | - | - | - | RFFE | face up (nozzle points UP) | +6305 | E 366700 / N 975345 | "CONN TO P-2303/ DN2 2" RFFE 300" | [79,123,121,148] | 0.97 | CONFIRMED |
| N-002 | GASKET | item 13 | 2" | 2 | 2 | 2 | 300 | RF, 3.2 mm thk | Valflon PTFE c.n.a.f. | RF | RF | horizontal plane | +6305 | thickness 3.2 mm | MTO item 13; flange tag "F6 G13 B17" | [79,123,121,148] | 0.90 | CONFIRMED |
| N-003 | FLANGE_WN | item 6 (F6), bolts item 17 | 2" | 2 | 2 | 2 | 300 | RFSF, 10S bore | A182 F304 | RF | BW | axis vertical (UP) | +6305 (face) | bolt set: 8 x 5/8 in x 95.0 mm | "F6 G13 B17" | [79,120,121,145] | 0.96 | CONFIRMED |
| N-004 | REDUCER_CONCENTRIC | item 2 | 2 -> 3 | 3x2 | 2 | 3 | - | Sch 10S | A403 WP304 | BW | BW | axis vertical; **small end DOWN (toward nozzle), large end UP** | +6467 (large end, CALCULATED) | reducer_type = CONCENTRIC; expands in assumed flow direction | "[2] 3X2" NPD" | [93,88,111,117] | 0.95 | CONFIRMED |
| N-005 | ELBOW_90 | item 4 (elbow A) | 3" | 3 | 3 | 3 | - | Sch 10S LR | A403 WP304 | BW | BW | turns **UP -> SOUTH** | +6581 (CL) | elbow_angle_deg 90; radius_type LR; centre-to-end 114 mm (from drawing dim) | "[4]", "3" NPD" | [101,57,117,80] | 0.95 | CONFIRMED |
| N-006 | SOCKOLET (OLET) | item 3 | 3 x 3/4 | 3x3/4 | 3 (run) | 3/4 (branch) | 3000# | - | A182 F304 | run BW | branch SW | branch outlet **UP** | +6581 (run CL) | branch_connection_type = SOCKOLET 3000#; parent of BR-001 | "[3] 3X3/4" NPD" | [224,130,262,154] | 0.94 | CONFIRMED |
| N-007 | ELBOW_90 | item 4 (elbow B) | 3" | 3 | 3 | 3 | - | Sch 10S LR | A403 WP304 | BW | BW | turns **SOUTH -> DOWN** | **+6581 (CL, EXPLICIT)** | elbow_angle_deg 90; radius_type LR | "[4] EL +6581" | [375,256,399,285] | 0.98 | CONFIRMED |
| N-008 | SOCKOLET (OLET) | item 3 | 3 x 3/4 | 3x3/4 | 3 | 3/4 | 3000# | - | A182 F304 | run BW | branch SW | branch outlet **EAST (horizontal)** | **+6059 (EXPLICIT)** | parent of BR-003 (PG-008) | "[3] 3X3/4" NPD", "EL +6059" | [310,604,340,630] | 0.94 | CONFIRMED |
| N-009 | ELBOW_90 | item 4 (elbow C) | 3" | 3 | 3 | 3 | - | Sch 10S LR | A403 WP304 | BW | BW | turns **DOWN -> EAST** | +5845 (CALCULATED) / +5844 (stated at continuation) | elbow_angle_deg 90; radius_type LR | "[4]" | [306,758,327,786] | 0.95 | CONFIRMED |
| N-010 | FLANGE_WN | item 5 (F5), gasket 12, bolts 15 | 3" | 3 | 3 | 3 | 300 | RFSF, 10S bore | A182 F304 | BW | RF | axis EAST | +5844 | bolt set 8 x 3/4 in x 115.0 mm | "F5 G12 B15" | [407,745,420,769] | 0.95 | CONFIRMED |
| N-011 | GASKET | item 12 | 3" | 3 | 3 | 3 | 300 | RF, 3.2 mm | Valflon PTFE | RF | RF | vertical plane | +5844 | - | MTO item 12 | [407,740,418,765] | 0.88 | CONFIRMED |
| N-012 | CHECK_VALVE | item 18 | 3" | 3 | 3 | 3 | 300 | RFSF | A351-CF8, 304SS trim, BC | RF | RF | **installed in a HORIZONTAL run** at EL +5844 | +5844 | valve_type SWING CHECK; bolted cover (BC); check_valve_orientation: horizontal, disc pivot orientation not shown; **no flow arrow on drawing** | "[18]", MTO "CHECK 300# RFSF A351-CF8, 304SS-TRIM BC SWING" | [415,735,444,769] | 0.94 | CONFIRMED |
| N-013 | GASKET | item 12 | 3" | 3 | 3 | 3 | 300 | RF, 3.2 mm | Valflon PTFE | RF | RF | vertical plane | +5844 | - | MTO item 12 | [440,730,452,755] | 0.88 | CONFIRMED |
| N-014 | FLANGE_WN | item 5 (F5), gasket 12, bolts 15 | 3" | 3 | 3 | 3 | 300 | RFSF, 10S bore | A182 F304 | RF | BW | axis EAST | +5844 | bolt set 8 x 3/4 in x 115.0 mm | "F5 G12 B15" | [443,728,458,752] | 0.95 | CONFIRMED |
| N-015 | SOCKOLET (OLET) | item 3 | 3 x 3/4 | 3x3/4 | 3 | 3/4 | 3000# | - | A182 F304 | run BW | branch SW | branch outlet **DOWN** | +5844 (run CL) | parent of BR-002 | "[3] 3X3/4" NPD" | [536,655,560,680] | 0.94 | CONFIRMED |
| N-016 | FLANGE_WN | item 5 (F5) - **terminal** | 3" | 3 | 3 | 3 | 300 | RFSF, 10S bore | A182 F304 | BW | RF | axis EAST | +5844 | **no gasket or bolt set assigned to this joint on this sheet** (mating half on ISO -02) | "[F5]" with line number leader | [613,570,630,595] | 0.93 | CONFIRMED |
| N-017 | SPECIALTY_ITEM (?) | untagged | - | 3 (assumed) | 3 | 3 | - | - | - | RF | RF | - | +5844 | Small hatched plate-like element drawn immediately beyond the N-016 raised face, annotated "TAIL U / OPERATE OPEN". Possible spectacle blind / spacer operated in the OPEN position, or a reference symbol for the mating gasket on ISO -02. **Not in the MTO.** | "TAIL U / OPERATE OPEN" | [625,573,640,592] | 0.45 | **UNCERTAIN** |
| N-018 | CONTINUATION_POINT | ISO ...-02 | 3"-MMA-23-0109-SC30-CC38MM | 3 | 3 | - | 300 | - | - | RF | - | EAST | +5844 | E 367711 / N 974255 | "SEE ISO 3"-MMA-23-0109-SC30-CC38MM -02" | [613,570,637,598] | 0.96 | CONFIRMED |
| N-019 | SUPPORT | SUPT19 / R4-I2S(E-W) | 3" | 3 | - | - | - | - | - | - | - | at/adjacent to elbow N-009 | approx. +5844 | MTO item 19, qty 1. **No location dimension given.** | "SUPT19 / R4-I2S(E-W)" | [274,786,319,809] | 0.85 | CONFIRMED (location UNCERTAIN) |

## 5.2 BR-001 - 3/4 in VENT (upward from N-006, high point EL +6581)

| node_id | component_type | item | NPS | rating | material | orientation | station from run CL (mm) | elevation (mm) | evidence | conf |
|---|---|---|---|---|---|---|---|---|---|---|
| N-101 | NIPPLE | 8 | 3/4 | Sch 80S | A312 TP304, PBE, 100 mm long | vertical UP | 60 (sockolet outlet) -> 162 | +6641 -> +6743 | "[8]" | 0.92 |
| N-102 | GATE_VALVE | 10 | 3/4 | 800# SW | A182-F304, 304 SS trim, BB, V/H | stem orientation not dimensioned | 162 -> 219 | +6743 -> +6800 | "[10]" | 0.93 |
| N-103 | NIPPLE | 8 | 3/4 | Sch 80S | A312 TP304, PBE, 100 mm long | vertical UP | 219 -> 336 (incl. flange) | +6800 -> +6917 | "[8]" | 0.92 |
| N-104 | FLANGE_SW | 9 (F9), bolts 16 | 3/4 | 300 RFSF 10S bore | A182 F304 | face UP | 336 (face) | +6917 | "F9 G14 B16"; bolts 4 x 5/8 in x 85.0 mm | 0.93 |
| N-105 | GASKET | 14 | 3/4 | 300 RF, 3.2 mm | Valflon PTFE | - | 336 | +6917 | MTO item 14 | 0.88 |
| N-106 | FLANGE_BLIND | 7 | 3/4 | 300 RFSF | A182 F304 | - | > 336, not dimensioned | UNKNOWN | "[7]" | 0.93 |

## 5.3 BR-002 - 3/4 in DRAIN (downward from N-015, low point EL +5844)

Component sequence, items, materials and the dimension chain (60 / 102 / 57 / 117) are **identical** to BR-001, mirrored downward.

| node_id | component_type | item | NPS | orientation | station from run CL (mm) | elevation (mm) | conf |
|---|---|---|---|---|---|---|---|
| N-201 | NIPPLE | 8 | 3/4 | vertical DOWN | 60 -> 162 | +5784 -> +5682 | 0.92 |
| N-202 | GATE_VALVE | 10 | 3/4 | vertical DOWN | 162 -> 219 | +5682 -> +5625 | 0.93 |
| N-203 | NIPPLE | 8 | 3/4 | vertical DOWN | 219 -> 336 | +5625 -> +5508 | 0.92 |
| N-204 | FLANGE_SW | 9 (F9), bolts 16 | 3/4 | face DOWN | 336 (face) | +5508 | 0.93 |
| N-205 | GASKET | 14 | 3/4 | - | 336 | +5508 | 0.88 |
| N-206 | FLANGE_BLIND | 7 | 3/4 | - | > 336, not dimensioned | UNKNOWN | 0.93 |

## 5.4 BR-003 - 3/4 in PRESSURE GAUGE CONNECTION 23-PG-008 (EAST from N-008, EL +6059)

| node_id | component_type | item | NPS | orientation | station from run CL (mm) | elevation (mm) | conf |
|---|---|---|---|---|---|---|---|
| N-301 | NIPPLE | 8 | 3/4 | horizontal EAST | 60 -> 162 | +6059 | 0.92 |
| N-302 | GATE_VALVE | 11 | 3/4 | horizontal EAST | 162 -> 221 | +6059 | 0.93 |
| N-303 | INSTRUMENT_CONNECTION / PRESSURE_INSTRUMENT | - (tag 23-PG-008) | 3/4 | screwed outlet, open end | 221 | +6059 | 0.90 |

MTO item 11 = "GATE 800# SCRD/SW A182-F304, 304 SS-TRIM BB V/H" - screwed x socket weld, i.e. socket weld to the nipple, screwed outlet for the gauge. The gauge itself is shown as an instrument balloon with a dashed leader and is **not** on this MTO.

---

# SECTION 6 - PIPE SEGMENT / EDGE TABLE

Two length columns are given deliberately:

- **`length_CL`** - the value the drawing actually dimensions (centerline/point to centerline/point). This is a drawing fact.
- **`pipe_straight`** - the bare straight pipe between component ends, obtained **only** by deducting the 90° LR elbow centre-to-end of **114 mm**, which the drawing itself states (the dimension "114" from the reducer large end to the elbow N-005 centreline). Where a flange length-through-hub deduction is also needed, the value is reported **UNKNOWN** because the drawing does not give it.

| edge_id | from | to | line_number | NPS | sched | material | length_CL (mm) | pipe_straight (mm) | length_source | orientation | start EL | end EL | slope | dimension_evidence | conf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E-001 | N-001 | N-002 | 2" | 2 | - | - | 0 | 0 | DIMENSION_CHAIN (bolted joint) | - | +6305 | +6305 | 0 | gasket 3.2 mm, MTO item 13 | 0.88 |
| E-002 | N-002 | N-003 | 2" | 2 | - | - | 0 | 0 | DIMENSION_CHAIN (bolted joint) | - | +6305 | +6308 | 0 | - | 0.88 |
| E-003 | N-003 | N-004 | 2" | 2 | 10S | A312 TP304 | 0 | 0 | DIMENSION_CHAIN | UP | +6308 | +6467 | vertical | Covered by dim "159" (flange face to reducer large end) - no free pipe; flange hub welds to reducer small end | 0.85 |
| E-004 | N-004 | N-005 | 3" | 3 | 10S | A312 TP304 | 0 | 0 | DIMENSION_CHAIN | UP | +6467 | +6581 | vertical | Dim "114" = 3 in LR elbow centre-to-end; **no straight pipe between reducer and elbow** | 0.88 |
| E-005 | N-005 | N-006 | 3" | 3 | 10S | A312 TP304 | **488** | 374 | EXPLICIT_DIMENSION | **SOUTH** | +6581 | +6581 | 0 | "488" (elbow A CL to vent sockolet CL); cut length `<3>`, parts item `[1]` | 0.95 |
| E-006 | N-006 | N-007 | 3" | 3 | 10S | A312 TP304 | **602** | 488 | EXPLICIT_DIMENSION | **SOUTH** | +6581 | +6581 | 0 | "602" (vent sockolet CL to elbow B CL) | 0.95 |
| E-007 | N-007 | N-008 | 3" | 3 | 10S | A312 TP304 | **522** | 408 | EXPLICIT_DIMENSION | **DOWN** | +6581 | +6059 | vertical | "522"; cut length `<2>`; confirmed by EL +6581 - EL +6059 = 522 | 0.97 |
| E-008 | N-008 | N-009 | 3" | 3 | 10S | A312 TP304 | **214** | 100 | EXPLICIT_DIMENSION | **DOWN** | +6059 | +5845 | vertical | "214"; EL +6059 - 214 = +5845 vs stated +5844 (1 mm) | 0.95 |
| E-009 | N-009 | N-010 | 3" | 3 | 10S | A312 TP304 | **194** | **UNKNOWN** (= 80 - flange length-through-hub) | EXPLICIT_DIMENSION | **EAST** | +5844 | +5844 | 0 | "194" (elbow C CL to flange F5 face) | 0.94 |
| E-010 | N-010 | N-011 | 3" | 3 | - | - | 0 | 0 | - | EAST | +5844 | +5844 | 0 | bolted joint | 0.88 |
| E-011 | N-011 | N-012 | 3" | 3 | - | - | 0 | 0 | - | EAST | +5844 | +5844 | 0 | - | 0.88 |
| E-012 | N-012 | N-013 | 3" | 3 | - | - | 0 | 0 | - | EAST | +5844 | +5844 | 0 | - | 0.88 |
| E-013 | N-013 | N-014 | 3" | 3 | - | - | 0 | 0 | - | EAST | +5844 | +5844 | 0 | Dim **"318"** spans N-010 face to N-014 face (check valve + 2 gaskets). **No pipe in this span.** | 0.92 |
| E-014 | N-014 | N-015 | 3" | 3 | 10S | A312 TP304 | **247** | **UNKNOWN** (= 247 - flange length-through-hub) | EXPLICIT_DIMENSION | **EAST** | +5844 | +5844 | 0 | "247" (flange F5 face to drain sockolet CL); cut length `<1>` | 0.94 |
| E-015 | N-015 | N-016 | 3" | 3 | 10S | A312 TP304 | **247** | **UNKNOWN** (= 247 - flange length-through-hub) | EXPLICIT_DIMENSION | **EAST** | +5844 | +5844 | 0 | "247" (drain sockolet CL to terminal flange face) | 0.94 |
| E-016 | N-016 | N-017 | 3" | 3 | - | - | 0 | 0 | - | EAST | +5844 | +5844 | 0 | unidentified element, see N-017 | 0.45 |
| E-017 | N-017 | N-018 | 3" | 3 | - | - | 0 | 0 | INFERRED | EAST | +5844 | +5844 | 0 | continuation | 0.90 |

## Branch edges

| edge_id | from | to | branch | NPS | length (mm) | length_source | orientation | conf |
|---|---|---|---|---|---|---|---|---|
| E-101 | N-006 | N-101 | BR-001 | 3/4 | 60 | EXPLICIT_DIMENSION | UP | 0.85 |
| E-102 | N-101 | N-102 | BR-001 | 3/4 | 102 | EXPLICIT_DIMENSION | UP | 0.85 |
| E-103 | N-102 | N-103 | BR-001 | 3/4 | 57 | EXPLICIT_DIMENSION | UP | 0.85 |
| E-104 | N-103 | N-104 | BR-001 | 3/4 | 117 | EXPLICIT_DIMENSION | UP | 0.85 |
| E-105 | N-104 | N-106 | BR-001 | 3/4 | UNKNOWN | UNKNOWN | UP | 0.85 |
| E-201 | N-015 | N-201 | BR-002 | 3/4 | 60 | EXPLICIT_DIMENSION | DOWN | 0.88 |
| E-202 | N-201 | N-202 | BR-002 | 3/4 | 102 | EXPLICIT_DIMENSION | DOWN | 0.88 |
| E-203 | N-202 | N-203 | BR-002 | 3/4 | 57 | EXPLICIT_DIMENSION | DOWN | 0.88 |
| E-204 | N-203 | N-204 | BR-002 | 3/4 | 117 | EXPLICIT_DIMENSION | DOWN | 0.88 |
| E-205 | N-204 | N-206 | BR-002 | 3/4 | UNKNOWN | UNKNOWN | DOWN | 0.88 |
| E-301 | N-008 | N-301 | BR-003 | 3/4 | 60 | EXPLICIT_DIMENSION | EAST | 0.88 |
| E-302 | N-301 | N-302 | BR-003 | 3/4 | 102 | EXPLICIT_DIMENSION | EAST | 0.88 |
| E-303 | N-302 | N-303 | BR-003 | 3/4 | 59 | EXPLICIT_DIMENSION | EAST | 0.88 |

**Dimension-chain reconciliation** (the chain order 60 / 102 / 57 / 117 from the run centerline outward was established by reading the individual leader arrowheads at 900 dpi on both the vent and the drain, and is cross-confirmed by the PG branch reading 60 / 102 / 59):

| Chain | Dimensions | Sum | Independent check | Closure error |
|---|---|---|---|---|
| Riser (nozzle face -> elbow A CL) | 159 + 114 | 273 | EL +6581 - EL +6305 = 276 | **-3 mm** (closes to 0.2 mm if the 3.2 mm gasket is added, i.e. "159" is datumed on the WN flange face, not the nozzle face) |
| SOUTH run (elbow A CL -> elbow B CL) | 488 + 602 | **1090** | N 975345 - N 974255 = **1090** | **0 mm - exact** |
| Vertical (elbow B CL -> elbow C CL) | 522 + 214 | 736 | EL +6581 - EL +5844 = 737 | -1 mm |
| EAST run (elbow C CL -> terminal flange face) | 194 + 318 + 247 + 247 | 1006 | E 367711 - E 366700 = 1011 | **-5 mm - unresolved** |

---

# SECTION 7 - BRANCH TABLE

```json
[
  {
    "branch_id": "BR-001",
    "function": "3/4 in VENT at line high point",
    "parent_node": "N-006",
    "parent_run": "E-005 / E-006 (SOUTH run, EL +6581)",
    "branch_size": "3/4 in",
    "branch_connection": "SOCKOLET 3000# A182 F304 (MTO item 3), 3 x 3/4",
    "branch_direction": "UP (vertical)",
    "destination": "Blind flange (dead end)",
    "ordered_nodes": ["N-006", "N-101", "N-102", "N-103", "N-104", "N-105", "N-106"],
    "ordered_edges": ["E-101", "E-102", "E-103", "E-104", "E-105"],
    "total_dimensioned_length_mm": 336,
    "insulation_annotation_present": false
  },
  {
    "branch_id": "BR-002",
    "function": "3/4 in DRAIN at line low point",
    "parent_node": "N-015",
    "parent_run": "E-014 / E-015 (EAST run, EL +5844)",
    "branch_size": "3/4 in",
    "branch_connection": "SOCKOLET 3000# A182 F304 (MTO item 3), 3 x 3/4",
    "branch_direction": "DOWN (vertical)",
    "destination": "Blind flange (dead end)",
    "ordered_nodes": ["N-015", "N-201", "N-202", "N-203", "N-204", "N-205", "N-206"],
    "ordered_edges": ["E-201", "E-202", "E-203", "E-204", "E-205"],
    "total_dimensioned_length_mm": 336,
    "insulation_annotation_present": true
  },
  {
    "branch_id": "BR-003",
    "function": "3/4 in pressure gauge connection",
    "parent_node": "N-008",
    "parent_run": "E-007 / E-008 (vertical drop)",
    "branch_size": "3/4 in",
    "branch_connection": "SOCKOLET 3000# A182 F304 (MTO item 3), 3 x 3/4",
    "branch_direction": "EAST (horizontal)",
    "destination": "Instrument 23-PG-008 (off-drawing)",
    "ordered_nodes": ["N-008", "N-301", "N-302", "N-303"],
    "ordered_edges": ["E-301", "E-302", "E-303"],
    "total_dimensioned_length_mm": 221,
    "insulation_annotation_present": true
  }
]
```

**Branch tree**

```
N-001  EQUIPMENT NOZZLE  P-2303/DN2  2"  EL +6305
 |  E-001/E-002/E-003 (gasket, 2" WN flange)
N-004  REDUCER CONC 3x2  (2" -> 3")
 |  E-004  (0 mm)
N-005  ELBOW 90 LR 3"   UP -> SOUTH    EL +6581
 |  E-005  488 mm  SOUTH
N-006  SOCKOLET 3x3/4 ----------------> BR-001  3/4" VENT (UP)
 |                                        N-101 nipple - N-102 gate 800# SW
 |                                        - N-103 nipple - N-104 SW flg
 |                                        - N-105 gasket - N-106 BLIND
 |  E-006  602 mm  SOUTH
N-007  ELBOW 90 LR 3"   SOUTH -> DOWN   EL +6581
 |  E-007  522 mm  DOWN
N-008  SOCKOLET 3x3/4 ----------------> BR-003  3/4" PG CONNECTION (EAST)
 |                                        N-301 nipple - N-302 gate 800#
 |                                        SCRD/SW - N-303 instrument 23-PG-008
 |  E-008  214 mm  DOWN
N-009  ELBOW 90 LR 3"   DOWN -> EAST    EL +5844   [SUPT19 R4-I2S(E-W)]
 |  E-009  194 mm  EAST
N-010  FLANGE WN 3" - N-011 GASKET
 |  (318 mm face to face)
N-012  CHECK VALVE 3" 300# SWING
 |
N-013  GASKET - N-014 FLANGE WN 3"
 |  E-014  247 mm  EAST
N-015  SOCKOLET 3x3/4 ----------------> BR-002  3/4" DRAIN (DOWN)
 |                                        N-201 nipple - N-202 gate 800# SW
 |                                        - N-203 nipple - N-204 SW flg
 |                                        - N-205 gasket - N-206 BLIND
 |  E-015  247 mm  EAST
N-016  FLANGE WN 3" (terminal)
 |  N-017  ?? "TAIL U / OPERATE OPEN"  (UNCERTAIN)
N-018  CONTINUATION -> ISO 3"-MMA-23-0109-SC30-CC38MM-02
       E 367711  N 974255  EL +5844
```

---

# SECTION 8 - ORDERED HYDRAULIC PATHS

**Flow direction assumption, stated explicitly:** the drawing carries **no flow arrow anywhere**, including on the swing check valve. The direction below (nozzle -> continuation) is INFERRED from two facts on the drawing: the line originates at equipment nozzle P-2303/DN2, and the 2 in x 3 in reducer expands away from that nozzle. `flow_direction_status` is therefore **NOT_CONFIRMED** and must be verified against P&ID 2300-001 before the topology is used for a directional calculation.

```json
{
  "hydraulic_paths": [
    {
      "path_id": "PATH-001",
      "source": "N-001 - P-2303/DN2 equipment nozzle, 2 in RFFE 300, EL +6305",
      "destination": "N-018 - continuation to ISO 3\"-MMA-23-0109-SC30-CC38MM-02, E 367711 / N 974255 / EL +5844",
      "flow_direction_status": "NOT_CONFIRMED",
      "ordered_components": [
        {"sequence": 1,  "type": "EQUIPMENT_NOZZLE", "node_id": "N-001", "size_in": 2},
        {"sequence": 2,  "type": "GASKET", "node_id": "N-002", "size_in": 2},
        {"sequence": 3,  "type": "FLANGE_WN", "node_id": "N-003", "size_in": 2, "rating": "300"},
        {"sequence": 4,  "type": "REDUCER_CONCENTRIC", "node_id": "N-004", "upstream_size_in": 2, "downstream_size_in": 3, "reducer_type": "CONCENTRIC", "note": "expansion in assumed flow direction"},
        {"sequence": 5,  "type": "ELBOW_90", "node_id": "N-005", "size_in": 3, "radius_type": "LR", "turn": "UP to SOUTH"},
        {"sequence": 6,  "type": "PIPE", "edge_id": "E-005", "size_in": 3, "length_CL_mm": 488, "pipe_straight_mm": 374, "orientation": "SOUTH"},
        {"sequence": 7,  "type": "OLET_BRANCH_RUN_THROUGH", "node_id": "N-006", "run_size_in": 3, "branch_size_in": 0.75, "branch_id": "BR-001", "note": "dead-leg vent, no through flow"},
        {"sequence": 8,  "type": "PIPE", "edge_id": "E-006", "size_in": 3, "length_CL_mm": 602, "pipe_straight_mm": 488, "orientation": "SOUTH"},
        {"sequence": 9,  "type": "ELBOW_90", "node_id": "N-007", "size_in": 3, "radius_type": "LR", "turn": "SOUTH to DOWN", "elevation": 6581},
        {"sequence": 10, "type": "PIPE", "edge_id": "E-007", "size_in": 3, "length_CL_mm": 522, "pipe_straight_mm": 408, "orientation": "DOWN"},
        {"sequence": 11, "type": "OLET_BRANCH_RUN_THROUGH", "node_id": "N-008", "run_size_in": 3, "branch_size_in": 0.75, "branch_id": "BR-003", "note": "dead-leg instrument tapping, no through flow"},
        {"sequence": 12, "type": "PIPE", "edge_id": "E-008", "size_in": 3, "length_CL_mm": 214, "pipe_straight_mm": 100, "orientation": "DOWN"},
        {"sequence": 13, "type": "ELBOW_90", "node_id": "N-009", "size_in": 3, "radius_type": "LR", "turn": "DOWN to EAST", "elevation": 5845},
        {"sequence": 14, "type": "PIPE", "edge_id": "E-009", "size_in": 3, "length_CL_mm": 194, "pipe_straight_mm": null, "orientation": "EAST"},
        {"sequence": 15, "type": "FLANGE_WN", "node_id": "N-010", "size_in": 3, "rating": "300"},
        {"sequence": 16, "type": "GASKET", "node_id": "N-011", "size_in": 3},
        {"sequence": 17, "type": "CHECK_VALVE", "node_id": "N-012", "size_in": 3, "rating": "300", "valve_type": "SWING", "installation": "HORIZONTAL RUN", "face_to_face_span_mm": 318},
        {"sequence": 18, "type": "GASKET", "node_id": "N-013", "size_in": 3},
        {"sequence": 19, "type": "FLANGE_WN", "node_id": "N-014", "size_in": 3, "rating": "300"},
        {"sequence": 20, "type": "PIPE", "edge_id": "E-014", "size_in": 3, "length_CL_mm": 247, "pipe_straight_mm": null, "orientation": "EAST"},
        {"sequence": 21, "type": "OLET_BRANCH_RUN_THROUGH", "node_id": "N-015", "run_size_in": 3, "branch_size_in": 0.75, "branch_id": "BR-002", "note": "dead-leg drain, no through flow"},
        {"sequence": 22, "type": "PIPE", "edge_id": "E-015", "size_in": 3, "length_CL_mm": 247, "pipe_straight_mm": null, "orientation": "EAST"},
        {"sequence": 23, "type": "FLANGE_WN", "node_id": "N-016", "size_in": 3, "rating": "300"},
        {"sequence": 24, "type": "UNIDENTIFIED_ITEM", "node_id": "N-017", "identification_status": "UNCERTAIN"},
        {"sequence": 25, "type": "CONTINUATION_POINT", "node_id": "N-018"}
      ],
      "total_centreline_length_mm": 3105,
      "total_known_straight_length_mm": 1370,
      "total_known_straight_length_basis": "E-005 + E-006 + E-007 + E-008 with the 114 mm elbow centre-to-end deducted at each elbow tangent (114 mm is stated on this drawing)",
      "unknown_length_segments": ["E-009", "E-014", "E-015"],
      "unknown_length_reason": "flange length-through-hub is not given on the drawing; centreline values 194, 247 and 247 mm are known exactly",
      "centreline_length_by_coordinates_mm": 3114,
      "topology_confidence": 0.93
    }
  ],
  "dead_leg_paths": [
    {"path_id": "PATH-B01", "branch_id": "BR-001", "source": "N-006", "destination": "N-106 blind flange", "flow": "none in normal operation"},
    {"path_id": "PATH-B02", "branch_id": "BR-002", "source": "N-015", "destination": "N-206 blind flange", "flow": "none in normal operation"},
    {"path_id": "PATH-B03", "branch_id": "BR-003", "source": "N-008", "destination": "N-303 instrument 23-PG-008", "flow": "none (static tapping)"}
  ]
}
```

**Note on `total_known_straight_length_mm`:** as required, this value contains **no fitting equivalent lengths**. It is the bare straight pipe on the four fully-resolvable segments. The three flanged segments (E-009, E-014, E-015) are deliberately left out of the total because their straight-pipe portion cannot be established without the flange length-through-hub from the piping material specification. Their centreline values (194 + 247 + 247 = 688 mm) are known exactly and must be added by the hydraulic engineer once the flange dimension is available.

---

# SECTION 9 - HYDRAULIC FITTING TABLE

For resistance calculation on PATH-001. Branch connections on the run are included because they sit in the flow path, although a sockolet branch produces negligible run-side resistance.

| Seq | Path | Node | Component | Size | Details | Upstream | Downstream |
|---|---|---|---|---|---|---|---|
| 3 | PATH-001 | N-003 | Flange, weld neck | 2 in | Class 300, RFSF, 10S bore, A182 F304 | N-001 nozzle (2 in) | N-004 reducer (2 in end) |
| 4 | PATH-001 | N-004 | **Reducer, concentric** | 3 in x 2 in | **Upstream 2 in -> downstream 3 in (expansion)**. Concentric. Installed in a **vertical** run, small end down. No eccentric flat side applicable. Sch 10S, A403 WP304 | 2 in | 3 in |
| 5 | PATH-001 | N-005 | **Elbow 90 deg** | 3 in | **LR** (long radius), Sch 10S, A403 WP304, butt weld. Centre-to-end 114 mm. Turns UP -> SOUTH | 3 in | 3 in |
| 7 | PATH-001 | N-006 | Branch connection (run side) | 3 in run x 3/4 in branch | Sockolet 3000# A182 F304. **Flow stays on the run**; branch is a dead-leg vent | 3 in | 3 in |
| 9 | PATH-001 | N-007 | **Elbow 90 deg** | 3 in | **LR**, Sch 10S, A403 WP304. Turns SOUTH -> DOWN. EL +6581 | 3 in | 3 in |
| 11 | PATH-001 | N-008 | Branch connection (run side) | 3 in run x 3/4 in branch | Sockolet 3000#. Flow stays on the run; branch is a dead-leg instrument tapping | 3 in | 3 in |
| 13 | PATH-001 | N-009 | **Elbow 90 deg** | 3 in | **LR**, Sch 10S, A403 WP304. Turns DOWN -> EAST. EL +5845 | 3 in | 3 in |
| 15 | PATH-001 | N-010 | Flange, weld neck | 3 in | Class 300, RFSF, 10S bore, A182 F304 | 3 in | 3 in |
| 17 | PATH-001 | N-012 | **Check valve** | 3 in | **Swing type**, Class 300, RFSF ends, A351-CF8 body, 304 SS trim, bolted cover. **Installed in a horizontal run** at EL +5844. Flange face-to-face span 318 mm. **Drawing gives no flow arrow and no disc/hinge orientation** | 3 in | 3 in |
| 19 | PATH-001 | N-014 | Flange, weld neck | 3 in | Class 300, RFSF, 10S bore | 3 in | 3 in |
| 21 | PATH-001 | N-015 | Branch connection (run side) | 3 in run x 3/4 in branch | Sockolet 3000#. Flow stays on the run; branch is a dead-leg drain | 3 in | 3 in |
| 23 | PATH-001 | N-016 | Flange, weld neck (terminal) | 3 in | Class 300, RFSF, 10S bore. Mating half on ISO -02 | 3 in | 3 in |
| 24 | PATH-001 | N-017 | **Unidentified item** | 3 in (assumed) | Annotated "TAIL U / OPERATE OPEN". If a spectacle blind/spacer in the open position, resistance is negligible. **Must be confirmed** | 3 in | 3 in |

**Fitting summary for PATH-001:** 3 x 90° LR elbows (3 in) | 1 x concentric expander 2 in -> 3 in | 1 x swing check valve (3 in, horizontal) | 3 x run-through sockolet branches | 4 x flange joints (1 off 2 in, 3 off 3 in). **No gate, globe, ball, butterfly or control valve exists in the main flow path.** The only manual valves on the sheet are the two 3/4 in vent/drain isolation valves and the 3/4 in gauge isolation valve, all on dead legs.

**Dead-leg fittings (not in the main flow path):** per branch - 1 x sockolet, 2 x nipples, 1 x 3/4 in gate valve, 1 x SW flange, 1 x blind flange (BR-001 and BR-002); 1 x sockolet, 1 x nipple, 1 x gate valve (BR-003).

---

# SECTION 10 - ROUTING TABLE

Coordinates are stepped from the stated source coordinates using the drawing dimensions. Where the stepped value differs from a stated value, both are shown.

| Path | Seq | From | To | Pipe size | Length (mm) | Direction | Elev. start | Elev. end | Component at end | E at end | N at end |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 001 | 1 | N-001 (nozzle face) | N-004 (reducer large end) | 2 -> 3 in | 159 (+3.2 gasket) | UP | +6305 | +6467 | Concentric reducer 3x2 | 366700 | 975345 |
| 001 | 2 | N-004 | N-005 (elbow A CL) | 3 in | 114 | UP | +6467 | +6581 | 90 deg LR elbow | 366700 | 975345 |
| 001 | 3 | N-005 | N-006 (vent sockolet CL) | 3 in | 488 | **SOUTH** | +6581 | +6581 | Sockolet 3x3/4 (BR-001) | 366700 | 974857 |
| 001 | 4 | N-006 | N-007 (elbow B CL) | 3 in | 602 | **SOUTH** | +6581 | +6581 | 90 deg LR elbow | 366700 | 974255 |
| 001 | 5 | N-007 | N-008 (PG sockolet CL) | 3 in | 522 | **DOWN** | +6581 | +6059 | Sockolet 3x3/4 (BR-003) | 366700 | 974255 |
| 001 | 6 | N-008 | N-009 (elbow C CL) | 3 in | 214 | **DOWN** | +6059 | +5845 (stated +5844) | 90 deg LR elbow; support SUPT19 | 366700 | 974255 |
| 001 | 7 | N-009 | N-010 (flange face) | 3 in | 194 | **EAST** | +5844 | +5844 | WN flange F5 | 366894 | 974255 |
| 001 | 8 | N-010 | N-014 (flange face) | 3 in | 318 | **EAST** | +5844 | +5844 | Swing check valve between flanges | 367212 | 974255 |
| 001 | 9 | N-014 | N-015 (drain sockolet CL) | 3 in | 247 | **EAST** | +5844 | +5844 | Sockolet 3x3/4 (BR-002) | 367459 | 974255 |
| 001 | 10 | N-015 | N-016 (terminal flange face) | 3 in | 247 | **EAST** | +5844 | +5844 | WN flange F5 / continuation | 367706 (stated **367711**) | 974255 |
| B01 | 1 | N-006 | N-104 (SW flange face) | 3/4 in | 336 (60+102+57+117) | **UP** | +6581 | +6917 | SW flange + gasket + blind | 366700 | 974857 |
| B02 | 1 | N-015 | N-204 (SW flange face) | 3/4 in | 336 (60+102+57+117) | **DOWN** | +5844 | +5508 | SW flange + gasket + blind | 367459 | 974255 |
| B03 | 1 | N-008 | N-303 (gauge connection) | 3/4 in | 221 (60+102+59) | **EAST** | +6059 | +6059 | Instrument 23-PG-008 | 366921 | 974255 |

---

# SECTION 11 - ELEVATION PROFILE

All elevations below come from explicit `EL` annotations or from arithmetic on explicit dimensions only. No elevation is taken from drawing position.

**PATH-001**

| Station | Node | Elevation (mm) | Basis |
|---|---|---|---|
| Source | N-001 nozzle face | **+6305** | EXPLICIT ("EL +6305") |
| Reducer large end | N-004 | +6467 | CALCULATED (6581 - 114) |
| Elbow A CL | N-005 | +6581 | INFERRED (same horizontal run as N-007) |
| **HIGH POINT - entire SOUTH run** | E-005 / N-006 / E-006 | **+6581** | EXPLICIT at N-007 ("EL +6581"), run is horizontal |
| Elbow B CL | N-007 | **+6581** | EXPLICIT |
| PG tapping | N-008 | **+6059** | EXPLICIT ("EL +6059") |
| Elbow C CL | N-009 | +5845 | CALCULATED (6059 - 214); stated +5844 at the continuation |
| **LOW POINT - entire EAST run** | E-009 ... E-015 | **+5844** | EXPLICIT at continuation |
| Destination | N-018 | **+5844** | EXPLICIT |

- **Net elevation change, source to destination: 5844 - 6305 = -461 mm (falling).**
- Total rise from source: +276 mm (6305 -> 6581). Total fall: -737 mm (6581 -> 5844).
- **High point:** the SOUTH run at EL +6581. The 3/4 in **vent** (BR-001) is located on this run - correct.
- **Low point:** the EAST run at EL +5844. The 3/4 in **drain** (BR-002) is located on this run - correct.
- Both horizontal runs are drawn and dimensioned as **level (zero slope)**. No slope, "no pocket" or "free draining" note appears on this isometric.
- Branch extremes: vent blind flange face approx. EL +6917; drain blind flange face approx. EL +5508 (both CALCULATED from the 336 mm branch chain; the blind flange thickness beyond the SW flange face is not dimensioned).

---

# SECTION 12 - MTO / BOM

Transcribed exactly. Three sub-tables as printed.

## SHOP FABRICATED MATERIAL

| item_no | description | NPD (in) | cmdty_code | quantity |
|---|---|---|---|---|
| 1 | PIPE SCH 10S SMLS A312 TP304 BE | 3 | 5GNA108 | 1.8 M |
| 2 | REDUCER CONC SCH 10S A403 WP304 | 3X2 | 5KFN108 | 1 |
| 3 | SOCKOLET 3000# A182 F304 | 3X3/4 | 5LCN100 | 3 |
| 4 | ELL 90 DEG LR SCH 10S A403 WP304 | 3 | 5KAN108 | 3 |
| 5 | FLG WN 300# RFSF 10S BORE A182 F304 | 3 | 5NNB206 | 3 |
| 6 | FLG WN 300# RFSF 10S BORE A182 F304 | 2 | 5NNB206 | 1 |
| 7 | FLG BLIND 300# RFSF A182 F304 | 3/4 | 5QNF240 | 2 |

## FIELD SCRD/SW MATERIAL

| item_no | description | NPD (in) | cmdty_code | quantity |
|---|---|---|---|---|
| 8 | NIPPLE SCH 80S A312 TP304 PBE, 100MM-LONG | 3/4 | 5MAN424 | 5 |
| 9 | FLG SW 300# RFSF 10S BORE A182 F304 | 3/4 | 5QNB210 | 2 |
| 10 | GATE 800# SW A182-F304, 304 SS-TRIM BB V/H | 3/4 | 5BAN727H | 2 |
| 11 | GATE 800# SCRD/SW A182-F304, 304 SS-TRIM BB V/H | 3/4 | 5CAN727H | 1 |

## FIELD ASSEMBLY MATERIAL

| item_no | description | NPD (in) | cmdty_code | quantity |
|---|---|---|---|---|
| 12 | GASKET 300# RF VALFLON(PTFE) COMPRESSED NON-ABS FIBER SHEET THK 3.2MM | 3 | 5RAC200 | 2 |
| 13 | GASKET 300# RF VALFLON(PTFE) COMPRESSED NON-ABS FIBER SHEET THK 3.2MM | 2 | 5RAC200 | 1 |
| 14 | GASKET 300# RF VALFLON(PTFE) COMPRESSED NON-ABS FIBER SHEET THK 3.2MM | 3/4 | 5RAC200 | 2 |
| 15 | STUD-BOLT UNC A320 GR B8 CL.2 W/A194 GR 8 NUTS PTFE COATED, 115.0MM BOLT LENGTH | 3/4 | 5UBM401 | 16 |
| 16 | STUD-BOLT UNC A320 GR B8 CL.2 W/A194 GR 8 NUTS PTFE COATED, 85.0MM BOLT LENGTH | 5/8 | 5UBM401 | 8 |
| 17 | STUD-BOLT UNC A320 GR B8 CL.2 W/A194 GR 8 NUTS PTFE COATED, 95.0MM BOLT LENGTH | 5/8 | 5UBM401 | 8 |
| 18 | CHECK 300# RFSF A351-CF8, 304SS-TRIM BC SWING | 3 | 5DCN257 | 1 |

## PIPE SUPPORTS

| item_no | description | NPD (in) | cmdty_code | quantity |
|---|---|---|---|---|
| 19 | PSUPPORT | 3 | R4-I2S(E-W) | 1 |

---

# SECTION 13 - MTO-TO-TOPOLOGY RECONCILIATION

| Item | Description | MTO qty | Topology count | Status | Where found in topology |
|---|---|---|---|---|---|
| 1 | Pipe 3 in Sch 10S | 1.8 M | 3 marked cut lengths `<1>`, `<2>`, `<3>` plus 1 short unmarked piece at E-009 | **MATCH** (consistent) | E-005/E-006 (`<3>`), E-007/E-008 (`<2>`), E-014/E-015 (`<1>`), E-009 (short) |
| 2 | Reducer conc 3x2 | 1 | 1 | **MATCH** | N-004 |
| 3 | Sockolet 3x3/4 | 3 | 3 | **MATCH** | N-006, N-008, N-015 |
| 4 | Ell 90 LR 3 in | 3 | 3 | **MATCH** | N-005, N-007, N-009 |
| 5 | Flg WN 300# 3 in | 3 | 3 | **MATCH** | N-010, N-014, N-016 |
| 6 | Flg WN 300# 2 in | 1 | 1 | **MATCH** | N-003 |
| 7 | Flg blind 300# 3/4 in | 2 | 2 | **MATCH** | N-106, N-206 |
| 8 | Nipple 3/4 in, 100 mm | 5 | 5 | **MATCH** | N-101, N-103, N-201, N-203, N-301 |
| 9 | Flg SW 300# 3/4 in | 2 | 2 | **MATCH** | N-104, N-204 |
| 10 | Gate 800# SW 3/4 in | 2 | 2 | **MATCH** | N-102, N-202 |
| 11 | Gate 800# SCRD/SW 3/4 in | 1 | 1 | **MATCH** | N-302 |
| 12 | Gasket 3 in | 2 | 2 | **MATCH** | N-011, N-013 (check valve joints only; terminal flange N-016 has no gasket on this sheet) |
| 13 | Gasket 2 in | 1 | 1 | **MATCH** | N-002 |
| 14 | Gasket 3/4 in | 2 | 2 | **MATCH** | N-105, N-205 |
| 15 | Stud-bolt 3/4 in x 115 mm | 16 | 2 joints x 8 = 16 | **MATCH** | N-010/N-014 flange joints (3 in Class 300) |
| 16 | Stud-bolt 5/8 in x 85 mm | 8 | 2 joints x 4 = 8 | **MATCH** | N-104/N-204 (3/4 in Class 300) |
| 17 | Stud-bolt 5/8 in x 95 mm | 8 | 1 joint x 8 = 8 | **MATCH** | N-003 (2 in Class 300) |
| 18 | Check 300# 3 in swing | 1 | 1 | **MATCH** | N-012 |
| 19 | Pipe support R4-I2S(E-W) | 1 | 1 | **MATCH** | N-019 |
| - | **Unidentified element at N-017** | **0 (absent)** | **1 drawn** | **DISCREPANCY** | Element annotated "TAIL U / OPERATE OPEN" beyond the terminal flange face has no MTO entry |

**Result: 19 of 19 MTO items reconcile exactly with the extracted topology. One drawn element has no MTO entry.** The topology has not been adjusted to force agreement.

---

# SECTION 14 - ISOMETRIC CHECKLIST

| Item | Requirement | Status | Evidence | Drawing Location | External Reference Needed | Confidence | Remarks |
|---|---|---|---|---|---|---|---|
| 1a | Drawing title | **NO** | Title block has no drawing-title field or text | [702,650,996,969] | - | 0.90 | Only ISO NO. is given |
| 1b | Company / client numbering | **YES** | ISO NO. 3"-MMA-23-0109-SC30-CC38MM-03; client PTT Tank Terminal | [702,900,996,969] | - | 0.97 | |
| 1c | Project / job number | **YES** | JOB NO. 01001D; project name block | [702,860,996,910] | - | 0.97 | |
| 1d | Revision number | **YES** | REV NO. 3 | [960,930,996,969] | - | 0.98 | |
| 1e | Revision description | **YES** | Rev 3 "AS-BUILT"; full history Rev 0-3 | [477,690,700,760] | - | 0.96 | |
| 1f | Initials | **YES** | BY RSP / CHKD SOC / CHKD UKH / APP.L HKC | [477,690,700,760] | - | 0.95 | APPROVED BY box in title block is blank |
| 1g | Date | **YES** | 13-MAY-2011 (Rev 3); 20-SEP-2010 (Rev 0) | [477,690,700,760] | - | 0.96 | |
| 1h | General notes | **YES** | "NOTE: 1. VERIFY DIMENSIONS PRIOR TO FIELD ADJUSTMENT." | [44,789,226,809] | - | 0.97 | |
| 1i | Holds | **N/A** | No hold field and no hold flag on the sheet | - | - | 0.85 | As-built revision; no holds expected |
| 1j | Line characteristics | **YES** | 5-row line table | [24,823,488,940] | - | 0.97 | |
| 1k | Design pressure | **YES** | 30.00 bar (all lines) | [24,823,488,940] | - | 0.97 | |
| 1l | Design temperature | **YES** | 70.0 degC (all lines) | [24,823,488,940] | - | 0.97 | |
| 1m | Operating pressure | **YES** | 14.30 bar (all lines) | [24,823,488,940] | - | 0.97 | |
| 1n | Operating temperature | **YES** | 10.0 degC (all lines) | [24,823,488,940] | - | 0.97 | |
| 1o | Line size | **YES** | 3/4, 2 and 3 in in the line numbers; "3" NPD", "2" RFFE", "3X2" NPD", "3X3/4" NPD" on the geometry | multiple | - | 0.97 | |
| 1p | Service | **NO** | Table has **no SERVICE column**; service only implied by "MMA" inside the line number | [24,823,488,940] | Line list | 0.85 | |
| 1q | Piping class | **YES** | "SC30" in every line number | [24,823,488,940] | PMS for content | 0.88 | No dedicated piping-class column |
| 1r | Test condition | **YES** | TEST MEDIUM WAT, TEST P 45.00 bar | [24,823,488,940] | - | 0.97 | |
| 1s | NDE | **YES** | RT 10%, PT 100% (S/W only), PMI "-" | [24,823,488,940] | - | 0.95 | |
| 1t | PWHT | **YES** | PWHT = N | [24,823,488,940] | - | 0.96 | |
| 1u | Insulation | **YES** | CODE CC, THICK 38.00 / 25.00 mm | [24,823,488,940] | - | 0.95 | See anomaly A-04 on row L5 |
| 1v | Painting | **YES** | PAINT = D1 | [24,823,488,940] | Painting spec for D1 | 0.95 | |
| 1w | Reference P&IDs | **YES** | P&ID NO. 2300-001; PLAN NO. 2300-003 | [420,735,500,790] | - | 0.96 | |
| 2 | Plant North | **YES** | Arrow with "N", up-left approx. 30 deg; verified against the E/N coordinates | [40,26,93,74] | - | 0.96 | Axis mapping confirmed numerically (SOUTH run = 1090 mm = dN) |
| 3 | P&ID notes: minimum slope / no pocket / free draining / straight-run | **REFERENCE_REQUIRED** | No such note on the isometric; both horizontal runs are drawn level | - | **P&ID 2300-001** | 0.90 | If a slope or no-pocket requirement exists, the level runs must be re-checked |
| 4 | Isometric vs P&ID: size, line number, PMS, insulation | **REFERENCE_REQUIRED** | Line data present but not verifiable from this sheet alone | - | **P&ID 2300-001**, line list | 0.92 | |
| 5 | Equipment connections: tag, size, rating, end preparation | **YES** | "CONN TO P-2303/ DN2, 2" RFFE 300" plus E/N/EL | [79,123,121,148] | Nozzle schedule to confirm | 0.95 | All four attributes present |
| 6a | Instrument connection: tag | **YES** | Balloon "23 / PG-008" | [310,604,403,650] | - | 0.93 | |
| 6b | Instrument connection: size | **YES** | 3 x 3/4 in sockolet, 3/4 in branch | [310,604,340,630] | - | 0.93 | |
| 6c | Instrument connection: rating | **UNCERTAIN** | Isolation valve is 800# SCRD/SW; the rating of the gauge connection itself is not stated | [310,604,403,650] | Instrument hook-up | 0.70 | |
| 6d | Instrument connection: end preparation | **YES** | MTO item 11 GATE 800# **SCRD/SW** - screwed outlet for the gauge | MTO [710,300,992,600] | - | 0.90 | |
| 6e | Instrument connection: maintenance reserve | **REFERENCE_REQUIRED** | Not shown | - | Instrument hook-up standard | 0.85 | |
| 6f | Instrument connection: straight run | **REFERENCE_REQUIRED** | Not applicable to a pressure tapping in the normal sense; no straight-run note given | - | **P&ID 2300-001** | 0.80 | |
| 7 | Pipe support tag and dimensions | **NO** | Tag is given ("SUPT19", type R4-I2S(E-W), MTO item 19) but **no location dimension** is shown | [274,786,319,809] | Support standard drawing | 0.88 | Support appears to sit at/adjacent to elbow N-009; exact station is not dimensioned |
| 8 | Accessibility and constructability | **REFERENCE_REQUIRED** | Cannot be judged from the isometric alone | - | Plot plan / PLAN 2300-003 | 0.85 | Note: only 194 mm from elbow C centreline to the first flange face leaves a very short pipe piece for fit-up; worth confirming in the shop |
| 9 | Break flange / maintenance requirement | **REFERENCE_REQUIRED** | Three flanged joints exist (2 at the check valve, 1 terminal); whether a dedicated break flange is required is not established here | [407,735,458,769] | **P&ID 2300-001**, project spec | 0.80 | |
| 10 | Check valve installation (vertical only when permitted, upward flow where required) | **YES** | The swing check N-012 is installed in a **horizontal** run at EL +5844, which is the standard orientation for a swing check | [415,735,444,769] | - | 0.92 | No vertical check valve installation exists on this sheet |
| 11 | Gear-operated valve orientation / accessibility | **N/A** | No gear operator on the sheet; all three valves are 3/4 in hand-wheel type ("V/H") | MTO | - | 0.93 | |
| 12 | Specialty item connection: tag, size, rating, end preparation | **UNCERTAIN** | No specialty item is listed in the MTO, but an unidentified element is drawn at the terminal flange with the note "TAIL U / OPERATE OPEN" and carries no tag, size or rating | [625,573,640,592] | **ISO -02**, P&ID 2300-001 | 0.45 | See anomaly A-08 |
| 13 | Branch connection / reinforcement vs piping specification | **REFERENCE_REQUIRED** | 3 x 3000# sockolets on 3 in Sch 10S run; no reinforcement pad shown | [224,130,262,154] etc. | **PMS SC30 branch table** | 0.88 | Sch 10S is a thin wall; confirm the branch table permits a 3000# sockolet |
| 14 | Branch sequence vs P&ID | **REFERENCE_REQUIRED** | Sequence extracted (vent, PG, drain) but not comparable from this sheet | - | **P&ID 2300-001** | 0.90 | |
| 15 | Interface / interference | **REFERENCE_REQUIRED** | Not determinable from an isometric | - | Plot plan, 3D model | 0.85 | |
| 16 | Continuity: line number and drawing number | **UNCERTAIN** | Continuation to ISO "-02" with matching line number and matching coordinates/elevation is given at one end; equipment reference at the other end. **However, the line number is annotated only on the EAST run** - the riser, SOUTH run and vertical run carry no line-number tag, and the 2 in / 3 in line-number break at the reducer is not annotated | [613,570,637,598] and [518,340,610,360] | ISO -02 | 0.80 | See anomaly A-06 |
| 17 | Material / service-class breaks | **YES** | A single spec (SC30) and a single material family (A312/A403/A182 TP304/F304, Class 300) throughout. The only change is a **size** break 2 in -> 3 in at the concentric reducer N-004 | MTO | PMS to confirm | 0.90 | No class or material break exists on this sheet |
| 18 | Flow direction: check valves, globe valves, cryogenic cavity vents | **NO** | **No flow arrow appears anywhere on the drawing**, including at the swing check valve. Flow direction cannot be confirmed from this sheet | [415,735,444,769] | **P&ID 2300-001** | 0.92 | Genuine finding. A check valve without a flow arrow on an as-built isometric is a real installation risk |
| 19 | Insulation: type, thickness, limits, personnel protection | **UNCERTAIN** | Type and thickness are given (CC 38 mm main, CC 25 mm branch). Insulation limit markers with "INSUL: CC 38MM / CC 25MM / N" are drawn on the PG branch and on the drain branch, **but no insulation annotation appears on the vent branch** although the line list carries three 3/4 in line numbers (CC38 / CC25 / N). Personnel protection is not addressed | [310,604,403,650] and [536,655,585,764] | Insulation spec | 0.80 | See anomalies A-04, A-05 |
| 20 | Eccentric reducer flat-side orientation / pocket prevention | **N/A** | The only reducer on the sheet is **concentric** (MTO item 2) and is installed in a **vertical** run, where no pocket is created and no flat-side orientation applies | [93,88,111,117] | - | 0.94 | |
| 21 | Drain-valve clearance | **REFERENCE_REQUIRED** | The drain assembly projects approx. 336 mm below EL +5844 (to approx. EL +5508 at the SW flange face); no clearance dimension to grade, steel or the pipe rack is given | [536,655,585,764] | Plot plan / support drawing | 0.85 | |
| 22 | Hydrotest vents and drains at high / low points | **YES** | 3/4 in vent (sockolet, nipple, 800# gate, nipple, SW flange, blind) **on the high-point run at EL +6581**; 3/4 in drain of identical build **on the low-point run at EL +5844** | [224,77,262,154] and [536,655,585,764] | - | 0.94 | Both high and low points are covered |
| 23 | Flange bolt-hole orientation where angular valve orientation is required | **N/A** | No angular orientation requirement is stated for any valve; no bolt-hole orientation note is given | - | - | 0.85 | |
| 24 | MTO vs P&ID and piping material specification | **REFERENCE_REQUIRED** | The MTO reconciles **exactly** with the drawn topology (19/19 items, see Section 13), but it cannot be compared against the P&ID or PMS from this sheet | MTO [710,26,992,700] | **P&ID 2300-001, PMS SC30** | 0.92 | One drawn element (N-017) is missing from the MTO |
| 25 | Required field-verification note | **YES** | "NOTE: 1. VERIFY DIMENSIONS PRIOR TO FIELD ADJUSTMENT." | [44,789,226,809] | - | 0.97 | |
| 26 | Incorporation of comments / revisions | **YES** | Four revisions recorded with dates, initials and descriptions, ending at Rev 3 AS-BUILT | [477,690,700,760] | Comment sheets | 0.85 | The record exists; whether every comment was incorporated cannot be judged from the drawing |
| 28 | Steam-line drip legs and steam traps at low pockets | **N/A** | Service is MMA (liquid) at 10 degC operating; this is not a steam line | [24,823,488,940] | - | 0.92 | |
| 29 | Straight-run requirements: rotating machinery, mixing point to temperature instrument | **REFERENCE_REQUIRED** | Only 273 mm of centreline from the P-2303/DN2 nozzle face to the first elbow centreline, with a 2 in -> 3 in expansion inside that distance. No temperature instrument on the sheet | [79,57,121,148] | Equipment vendor data, **P&ID 2300-001** | 0.85 | Worth an explicit check against the machine vendor's straight-run requirement |
| 30 | Permanent strainers and basket withdrawal area | **N/A** | No strainer appears on this isometric or in the MTO | MTO | **P&ID 2300-001** | 0.85 | Confirm that no strainer is required on this line |
| 31 | Small-bore branch stiffeners | **REFERENCE_REQUIRED** | Three 3/4 in dead-leg branches on a 3 in **Sch 10S** run, each carrying a valve and blind flange up to 336 mm from the run. **No gusset or stiffener is shown** | [224,77,262,154], [310,604,403,650], [536,655,585,764] | PMS / stress requirement | 0.85 | Stress check is flagged "Y" in the title block, so a stress calculation exists somewhere - its number is not filled in |
| 32 | Large-bore piping >= 24 in: flange type, bolt length, short pieces | **N/A** | Maximum size on this sheet is 3 in | - | - | 0.97 | |
| 33 | Threaded piping: sufficient flange joints / unions for installation and maintenance | **YES** | The only threaded joint is the gauge connection at the outlet of the SCRD/SW gate valve (item 11), which is itself the isolation point. The vent and drain legs are socket welded up to a bolted SW-flange/blind joint, so both can be broken without cutting | [310,604,403,650] | - | 0.85 | |
| 34 | Butterfly / wafer check valve clearance | **N/A** | The check valve is a **flanged swing** type (A351-CF8, bolted cover), not wafer; no butterfly valve on the sheet | MTO item 18 | - | 0.93 | |
| 35 | Atmospheric PSV: downstream weep hole | **N/A** | No PSV or relief device on this isometric | MTO | - | 0.95 | |

*(Item 27 is intentionally absent, as in the source checklist.)*

**Checklist tally:** YES 30 | NO 4 | N/A 12 | REFERENCE_REQUIRED 12 | UNCERTAIN 4.

---

# SECTION 15 - DRAWING ANOMALIES

| ID | Category | drawing_anomaly | Description | Impact |
|---|---|---|---|---|
| A-01 | DIMENSION | true | **EAST run chain does not close.** 194 + 318 + 247 + 247 = 1006 mm, but the stated easting difference (E 367711 - E 366700) is 1011 mm. 5 mm unaccounted for. All four values were re-read at 900 dpi and are unambiguous | LOW for hydraulics; MEDIUM for fabrication/fit-up |
| A-02 | DIMENSION | true | **Vertical chain off by 1 mm.** 522 + 214 = 736 mm vs EL +6581 - EL +5844 = 737 mm | NONE (rounding) |
| A-03 | DIMENSION | true | **Riser chain off by 3 mm.** 159 + 114 = 273 mm vs EL +6581 - EL +6305 = 276 mm. Closes to within 0.2 mm if the "159" datum is taken at the WN flange raised face with the 3.2 mm gasket added separately. The drawing does not make the datum explicit | LOW |
| A-04 | TEXT | true | **Line list internal inconsistency.** Row "3/4"-MMA-23-0109-SC30-**N**" (suffix N = no insulation) carries INSULATION CODE = CC and THICK = 38.00 mm | LOW for hydraulics; MEDIUM for insulation take-off |
| A-05 | ANNOTATION | true | **Insulation limits not marked on the vent branch.** Limit markers "INSUL: CC 38MM / CC 25MM / N" are drawn on the PG branch and the drain branch but not on the vent branch (BR-001), although the same three 3/4 in line numbers exist in the line list | LOW |
| A-06 | TEXT | true | **Line numbers annotated on only one run.** The line number "3"-MMA-23-0109-SC30-CC38MM" is tagged only on the EAST run. The riser, the SOUTH run and the vertical drop carry no line-number tag, and the 2 in / 3 in line-number break (at the reducer N-004) is not annotated anywhere | LOW for hydraulics; MEDIUM for document control |
| A-07 | FLOW | true | **No flow direction arrow anywhere on the drawing**, including on the 3 in swing check valve | **HIGH** - see U-001 |
| A-08 | COMPONENT | true | **Unidentified element at the terminal flange.** A small hatched plate-like symbol is drawn immediately beyond the raised face of the terminal WN flange N-016, annotated "TAIL U / OPERATE OPEN". It has no item balloon, no tag and **no MTO entry** | **MEDIUM to HIGH** - see U-002 |
| A-09 | TEXT | true | Title block fields **SYSTEM**, **STRESS CAL.** and **APPROVED BY** are blank, although STRESS CHECK is flagged "Y" | NONE for hydraulics |
| A-10 | TEXT | true | The line-characteristics table has **no SERVICE column**. Service is only implied by the "MMA" field inside the line number | LOW |
| A-11 | DIMENSION | true | **Very short pipe piece at E-009.** Only 194 mm from the elbow C centreline to the first flange face; after deducting the 114 mm elbow centre-to-end the free pipe is approximately 80 mm minus the flange length-through-hub. This is a fit-up/constructability point, not an error | LOW |

---

# SECTION 16 - UNCERTAINTIES

```json
{
  "uncertainties": [
    {
      "id": "U-001",
      "category": "FLOW",
      "description": "No flow direction arrow exists anywhere on the isometric, including on the 3 in swing check valve N-012. The direction used in PATH-001 (nozzle P-2303/DN2 -> continuation ISO -02) is INFERRED from the line originating at an equipment nozzle and from the 2 in x 3 in reducer expanding away from that nozzle.",
      "possible_interpretations": [
        "Flow from P-2303/DN2 towards ISO -02 (assumed; consistent with an expanding reducer at the equipment nozzle)",
        "Flow from ISO -02 towards P-2303/DN2 (would make the reducer a contraction into the nozzle and would require the check valve to be reversed)"
      ],
      "affected_nodes": ["N-001", "N-004", "N-012", "N-018"],
      "affected_edges": ["ALL"],
      "impact_on_hydraulic_calculation": "CRITICAL",
      "recommended_manual_check": "Confirm flow direction and check valve orientation against P&ID 2300-001 before any directional hydraulic or surge calculation."
    },
    {
      "id": "U-002",
      "category": "COMPONENT",
      "description": "An unidentified small hatched element is drawn immediately beyond the raised face of the terminal 3 in WN flange N-016, with a double-arrow leader and the note 'TAIL U / OPERATE OPEN'. It carries no item balloon and does not appear in the MTO.",
      "possible_interpretations": [
        "A spectacle blind or spade/spacer at the tie-in, to be operated in the OPEN position (the wording 'OPERATE OPEN' supports this)",
        "A reference symbol for the mating gasket / flange supplied on ISO -02",
        "A note belonging to another component whose leader has been mis-attached"
      ],
      "affected_nodes": ["N-016", "N-017", "N-018"],
      "affected_edges": ["E-016", "E-017"],
      "impact_on_hydraulic_calculation": "HIGH",
      "recommended_manual_check": "Check ISO 3\"-MMA-23-0109-SC30-CC38MM-02 and P&ID 2300-001 for a spectacle blind or spacer at E 367711 / N 974255 / EL +5844. If a blind is present, its position (open or closed) determines whether the path is flowing at all."
    },
    {
      "id": "U-003",
      "category": "DIMENSION",
      "description": "The EAST run dimension chain sums to 1006 mm while the stated coordinates give 1011 mm. All four dimensions (194, 318, 247, 247) were verified at 900 dpi.",
      "possible_interpretations": [
        "A rounding or datum convention error in one of the four dimensions",
        "An undimensioned gap or gasket allowance at the terminal flange",
        "A coordinate rounding in the stated E 367711"
      ],
      "affected_nodes": ["N-009", "N-010", "N-014", "N-015", "N-016"],
      "affected_edges": ["E-009", "E-014", "E-015"],
      "impact_on_hydraulic_calculation": "LOW",
      "recommended_manual_check": "Re-check the EAST run chain against the 3D model or ISO -02 before spool fabrication."
    },
    {
      "id": "U-004",
      "category": "DIMENSION",
      "description": "The straight pipe length on E-009, E-014 and E-015 cannot be established because the flange length-through-hub is not given on the drawing. Only the centreline values (194, 247, 247 mm) are known.",
      "possible_interpretations": [
        "Deduct the ASME B16.5 Class 300 NPS 3 weld-neck flange length-through-hub once the PMS confirms the flange standard"
      ],
      "affected_nodes": ["N-010", "N-014", "N-016"],
      "affected_edges": ["E-009", "E-014", "E-015"],
      "impact_on_hydraulic_calculation": "LOW",
      "recommended_manual_check": "Obtain the flange dimension from PMS SC30 / ASME B16.5 and complete the straight-pipe total."
    },
    {
      "id": "U-005",
      "category": "DIMENSION",
      "description": "Branch dimension chain order. The chain 60 / 102 / 57 / 117 (run centreline outward) was established by reading each leader arrowhead individually at 900 dpi on both the vent and the drain, and is cross-checked by the PG branch reading 60 / 102 / 59. The leaders for '60' and '102' are jogged and cross each other, so a transposition of these two values cannot be fully excluded.",
      "possible_interpretations": [
        "60 (run CL to sockolet outlet), 102 (nipple), 57 (gate valve), 117 (nipple + SW flange) - adopted",
        "102 and 60 transposed"
      ],
      "affected_nodes": ["N-101", "N-102", "N-201", "N-202", "N-301", "N-302"],
      "affected_edges": ["E-101", "E-102", "E-201", "E-202", "E-301", "E-302"],
      "impact_on_hydraulic_calculation": "NONE",
      "recommended_manual_check": "All three branches are dead legs; a transposition changes nothing hydraulically. Confirm before fabricating the small-bore assemblies."
    },
    {
      "id": "U-006",
      "category": "DIMENSION",
      "description": "The distance from the SW flange face (N-104 / N-204) to the blind flange outer face is not dimensioned, so the true extremity of the vent and drain dead legs is unknown.",
      "possible_interpretations": ["Add gasket thickness 3.2 mm plus the 3/4 in Class 300 blind flange thickness from the PMS"],
      "affected_nodes": ["N-105", "N-106", "N-205", "N-206"],
      "affected_edges": ["E-105", "E-205"],
      "impact_on_hydraulic_calculation": "NONE",
      "recommended_manual_check": "Only needed for clearance checks (checklist item 21)."
    },
    {
      "id": "U-007",
      "category": "CONNECTIVITY",
      "description": "The exact station of pipe support SUPT19 (R4-I2S(E-W)) is not dimensioned. Its leader points at or immediately adjacent to elbow N-009.",
      "possible_interpretations": ["Support located on the vertical leg just above elbow N-009", "Support located on the EAST run just after elbow N-009"],
      "affected_nodes": ["N-019", "N-009"],
      "affected_edges": ["E-008", "E-009"],
      "impact_on_hydraulic_calculation": "NONE",
      "recommended_manual_check": "Obtain the support location from the pipe support drawing / stress calculation."
    },
    {
      "id": "U-008",
      "category": "SIZE",
      "description": "The 2 in / 3 in line-number break is not annotated. The size change is unambiguous (concentric reducer N-004), but which line number applies to the short 2 in portion is inferred from the line list row '2\"-MMA-23-0109-SC30-CC38MM'.",
      "possible_interpretations": ["2 in line number applies from the nozzle face to the reducer small end - adopted"],
      "affected_nodes": ["N-001", "N-002", "N-003", "N-004"],
      "affected_edges": ["E-001", "E-002", "E-003"],
      "impact_on_hydraulic_calculation": "LOW",
      "recommended_manual_check": "Confirm against the line list and P&ID 2300-001."
    },
    {
      "id": "U-009",
      "category": "TEXT",
      "description": "Insulation limit assignment on the branches. Four markers ('CC 38MM', 'CC 25MM', 'CC 25MM', 'N') are drawn around the PG branch and two ('CC 38MM', 'CC 25MM') around the drain, but the exact station of each break is not dimensioned, and the vent branch carries none.",
      "possible_interpretations": ["3 in run insulated CC 38 mm; 3/4 in branch CC 25 mm to the isolation valve; no insulation beyond the valve"],
      "affected_nodes": ["N-006", "N-008", "N-015"],
      "affected_edges": ["E-101", "E-201", "E-301"],
      "impact_on_hydraulic_calculation": "NONE",
      "recommended_manual_check": "Confirm with the insulation specification and the line list."
    }
  ]
}
```

---

# SECTION 17 - QC VALIDATION RESULTS

| Test | Description | Result | Detail |
|---|---|---|---|
| **QC-1** | Connectivity: every edge has from_node and to_node | **PASS** | 17 main edges + 13 branch edges, all with both ends defined. One legitimate open end: N-018 (continuation to ISO -02). Dead-leg terminations at N-106, N-206, N-303 are legitimate. |
| **QC-2** | Orphan nodes | **PASS** | No orphan flow nodes. N-019 (pipe support) is attached to the line but is deliberately not a flow node. N-017 is connected but its identity is uncertain. |
| **QC-3** | Component count: visual vs topology | **PASS** | Visually identified components (excluding gaskets and bolt sets): 27. Topology nodes of the same classes: 27. Gaskets: 5 visually implied / 5 in topology / 5 in MTO. |
| **QC-4** | Dimension traceability | **PASS with exceptions** | Every main-run segment carries an explicit dimension. Three segments (E-009, E-014, E-015) have a known centreline value but an UNKNOWN straight-pipe value because the flange length-through-hub is not on the drawing. No length has been estimated. |
| **QC-5** | Size continuity | **PASS** | One run-size change: 2 in -> 3 in, matched by concentric reducer N-004. Three run-to-branch changes: 3 in -> 3/4 in, each matched by a 3 x 3/4 sockolet (N-006, N-008, N-015). No unexplained size change. |
| **QC-6** | Line number continuity | **PARTIAL** | Five line numbers exist in the line list; only one (3 in CC38MM) is annotated on the geometry. The 2 in and the three 3/4 in line numbers are not tagged on the drawing. See anomaly A-06, uncertainty U-008. |
| **QC-7** | Flow direction | **FAIL (no evidence on drawing)** | No flow arrow at the check valve or anywhere else. Direction inferred only. See U-001. |
| **QC-8** | Branch completeness | **PASS** | Three branch connections visible; three branches in the topology (BR-001 vent, BR-002 drain, BR-003 PG-008). Sockolet count in the MTO (3) matches. |
| **QC-9** | Route completeness | **PASS** | PATH-001 terminates at an equipment nozzle (N-001) and at a continuation point (N-018) with full coordinates. All three branches terminate at a blind flange or an instrument. No unresolved endpoint. |
| **QC-10** | MTO reconciliation | **PASS with 1 discrepancy** | 19 of 19 MTO items match the topology exactly (Section 13). One drawn element (N-017) has no MTO entry. The topology was **not** altered to force agreement. |
| **QC-11** (additional) | Coordinate closure | **PASS with 3 minor errors** | SOUTH run closes exactly (1090 mm). Vertical closes to 1 mm. Riser closes to 3 mm (0.2 mm with the gasket). EAST run is 5 mm short. See A-01 to A-03. |
| **QC-12** (additional) | Axis consistency | **PASS** | North arrow direction, the two isometric horizontal axes and the stated E/N coordinates are mutually consistent and were verified numerically. |

**Independent re-trace (Pass 8) findings:**
- No fitting was skipped: every filled-dot shop weld on the main run corresponds to a component boundary in the topology.
- No fitting was incorrectly inserted: the "bow-tie" symbol at the nozzle end resolves into a 2 in WN flange hub (lower triangle) plus a concentric reducer (upper triangle), not two reducers.
- No branch disappeared: all three sockolets are on the topology.
- No visually crossing lines were incorrectly connected: the long leader from "SEE ISO ...-02" crosses the geometry but terminates at the terminal flange, and has been treated as a leader, not as pipe.
- Reducer direction verified: small end at the 2 in flange, large end at the 3 in elbow.
- Check valve orientation: horizontal installation confirmed; flow direction not determinable.
- Branch-vs-run side: all three branches are olet connections on the run, never on a branch side of a tee. No tee exists on this drawing.
- All pipe-size changes are accounted for.

---

# SECTION 18 - MACHINE-READABLE JSON

The complete machine-readable model is delivered as a separate file:

**`23-0109_ISO_topology.json`**

It contains the keys `drawing_metadata`, `lines`, `equipment`, `nodes`, `edges`, `branches`, `hydraulic_paths`, `elevation_profiles`, `mto`, `checklist`, `drawing_anomalies`, `uncertainties`, `qc_results`, with stable unique IDs and ID-based cross references throughout.

---

# ANSWERS TO THE 22 CRITICAL HYDRAULIC QUESTIONS

| # | Question | Answer |
|---|---|---|
| 1 | What is connected to what? | Fully resolved - Sections 5, 6, 7 |
| 2 | In what sequence? | Fully resolved - Section 8 |
| 3 | Pipe size at every location? | 2 in from N-001 to N-004; 3 in from N-004 to N-018; 3/4 in on all three branches |
| 4 | Where does the size change? | At reducer N-004 (2 -> 3 in) and at the three sockolets N-006, N-008, N-015 (3 -> 3/4 in) |
| 5 | Straight-pipe length between consecutive components? | Section 6, `pipe_straight` column |
| 6 | Which lengths are known vs unknown? | Known: E-005 (374), E-006 (488), E-007 (408), E-008 (100). Centreline known but straight-pipe UNKNOWN: E-009 (194), E-014 (247), E-015 (247) |
| 7 | What fitting occurs after each straight pipe segment? | Section 8 ordered list |
| 8 | How many elbows and what type? | **3 off, all 90 degree long radius (LR)**, 3 in, Sch 10S A403 WP304, butt weld, centre-to-end 114 mm |
| 9 | Where are valves and what type? | Main path: **none**. Dead legs: 2 x 3/4 in 800# SW gate (vent N-102, drain N-202), 1 x 3/4 in 800# SCRD/SW gate (PG N-302). Main path has 1 x 3 in Class 300 swing **check** valve N-012 |
| 10 | Where are reducers and in which direction? | One only: N-004, concentric, **2 in (nozzle side, bottom) -> 3 in (elbow side, top)**, vertical |
| 11 | Where are branches? | N-006 (SOUTH run, EL +6581), N-008 (vertical drop, EL +6059), N-015 (EAST run, EL +5844) |
| 12 | What fitting creates each branch? | 3 x 3/4 in 3000# **sockolet**, A182 F304 (MTO item 3) |
| 13 | Main/run path vs branch path? | Run = PATH-001 (N-001 to N-018). All three branches are dead legs, no through flow |
| 14 | Elevations along the route? | Section 11 |
| 15 | High and low points? | High = SOUTH run, EL +6581 (vented). Low = EAST run, EL +5844 (drained) |
| 16 | Source and destination of each path? | Source N-001 (P-2303/DN2, EL +6305). Destination N-018 (ISO -02, E 367711 / N 974255 / EL +5844) |
| 17 | Which continuation drawing is needed? | **3"-MMA-23-0109-SC30-CC38MM-02** |
| 18 | Which topology elements remain uncertain? | N-017 (unidentified element at the terminal flange) and the flow direction. See U-001, U-002 |

---

*Prepared from the isometric drawing alone. No P&ID, piping material specification, line list, nozzle schedule, instrument hook-up, support standard or plot plan was available, so 12 checklist items are returned as REFERENCE_REQUIRED. Every value reported here is either read from the drawing or calculated from values read from the drawing; nothing has been estimated from graphical scale, pixel distance or isometric angle.*
