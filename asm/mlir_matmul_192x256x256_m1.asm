matmul/matmul_192x256x256:
(__TEXT,__text) section
_matmul:
000000010000304c	sub	sp, sp, #0xf0
0000000100003050	stp	d15, d14, [sp, #0x80]   ; Latency: 6
0000000100003054	stp	d13, d12, [sp, #0x90]   ; Latency: 6
0000000100003058	stp	d11, d10, [sp, #0xa0]   ; Latency: 6
000000010000305c	stp	d9, d8, [sp, #0xb0]     ; Latency: 6
0000000100003060	stp	x24, x23, [sp, #0xc0]   ; Latency: 6
0000000100003064	stp	x22, x21, [sp, #0xd0]   ; Latency: 6
0000000100003068	stp	x20, x19, [sp, #0xe0]   ; Latency: 6
000000010000306c	mov	x10, #0x0
0000000100003070	ldp	x9, x8, [sp, #0x148]    ; Latency: 4
0000000100003074	ldp	x12, x11, [sp, #0x128]  ; Latency: 4
0000000100003078	ldp	x0, x17, [sp, #0x110]   ; Latency: 4
000000010000307c	add	x13, x1, x2, lsl #2     ; Latency: 2
0000000100003080	add	x14, x5, x5, lsl #1     ; Latency: 2
0000000100003084	lsl	x14, x14, #3
0000000100003088	lsl	x15, x6, #6
000000010000308c	ldp	x1, x16, [sp, #0xf0]    ; Latency: 4
0000000100003090	add	x16, x1, x16, lsl #2    ; Latency: 2
0000000100003094	lsl	x17, x17, #6
0000000100003098	lsl	x0, x0, #6
000000010000309c	mov	w1, #0x1400
00000001000030a0	mov	x2, #0x0
00000001000030a4	madd	x3, x10, x9, x11        ; Latency: 5
00000001000030a8	mov	x4, x16                 ; Latency: 2
00000001000030ac	madd	x5, x2, x8, x3          ; Latency: 5
00000001000030b0	add	x5, x12, x5, lsl #2     ; Latency: 2
00000001000030b4	ldp	q0, q10, [x5, #0x20]    ; Latency: 4
00000001000030b8	str	q0, [sp, #0x30]         ; Latency: 4
00000001000030bc	ldp	q16, q0, [x5]           ; Latency: 4
00000001000030c0	str	q0, [sp, #0x60]         ; Latency: 4
00000001000030c4	add	x6, x5, #0x400
00000001000030c8	ldr	q2, [x5, #0x430]        ; Latency: 4
00000001000030cc	ldr	q3, [x5, #0x420]        ; Latency: 4
00000001000030d0	ldr	q7, [x5, #0x410]        ; Latency: 4
00000001000030d4	ldr	q20, [x5, #0x400]       ; Latency: 4
00000001000030d8	add	x7, x5, #0x800
00000001000030dc	ldr	q4, [x5, #0x830]        ; Latency: 4
00000001000030e0	ldr	q5, [x5, #0x820]        ; Latency: 4
00000001000030e4	ldr	q19, [x5, #0x810]       ; Latency: 4
00000001000030e8	ldr	q25, [x5, #0x800]       ; Latency: 4
00000001000030ec	add	x19, x5, #0xc00
00000001000030f0	ldr	q17, [x5, #0xc30]       ; Latency: 4
00000001000030f4	ldr	q18, [x5, #0xc20]       ; Latency: 4
00000001000030f8	ldr	q24, [x5, #0xc10]       ; Latency: 4
00000001000030fc	ldr	q27, [x5, #0xc00]       ; Latency: 4
0000000100003100	add	x21, x5, #0x1, lsl #12
0000000100003104	ldr	q22, [x5, #0x1030]      ; Latency: 4
0000000100003108	ldr	q23, [x5, #0x1020]      ; Latency: 4
000000010000310c	ldr	q26, [x5, #0x1010]      ; Latency: 4
0000000100003110	ldr	q29, [x5, #0x1000]      ; Latency: 4
0000000100003114	add	x20, x5, x1             ; Latency: 2
0000000100003118	ldr	q28, [x5, #0x1430]      ; Latency: 4
000000010000311c	ldr	q30, [x5, #0x1420]      ; Latency: 4
0000000100003120	ldr	q21, [x5, #0x1410]      ; Latency: 4
0000000100003124	mov	x22, #-0x10
0000000100003128	mov	x23, x4                 ; Latency: 2
000000010000312c	mov	x24, x13                ; Latency: 2
0000000100003130	ldr	q31, [x5, #0x1400]      ; Latency: 4
0000000100003134	ldr	q11, [x24]              ; Latency: 4
0000000100003138	ldp	q9, q14, [x23, #0x20]   ; Latency: 4
000000010000313c	fmla.4s	v10, v14, v11[0]        ; Latency: 2
0000000100003140	str	q10, [sp, #0x70]        ; Latency: 4
0000000100003144	ldr	q1, [sp, #0x30]         ; Latency: 4
0000000100003148	fmla.4s	v1, v9, v11[0]          ; Latency: 2
000000010000314c	ldp	q8, q0, [x23]           ; Latency: 4
0000000100003150	fmla.4s	v16, v8, v11[0]         ; Latency: 2
0000000100003154	mov.16b	v6, v16
0000000100003158	ldr	q10, [x24, #0x400]      ; Latency: 4
000000010000315c	fmla.4s	v2, v14, v10[0]         ; Latency: 2
0000000100003160	str	q2, [sp, #0x50]         ; Latency: 4
0000000100003164	fmla.4s	v3, v9, v10[0]          ; Latency: 2
0000000100003168	mov.16b	v12, v3
000000010000316c	fmla.4s	v20, v8, v10[0]         ; Latency: 2
0000000100003170	ldr	q2, [x24, #0x800]       ; Latency: 4
0000000100003174	fmla.4s	v4, v14, v2[0]          ; Latency: 2
0000000100003178	mov.16b	v16, v5
000000010000317c	fmla.4s	v16, v9, v2[0]          ; Latency: 2
0000000100003180	fmla.4s	v25, v8, v2[0]          ; Latency: 2
0000000100003184	ldr	q3, [x24, #0xc00]       ; Latency: 4
0000000100003188	fmla.4s	v17, v14, v3[0]         ; Latency: 2
000000010000318c	stp	q17, q4, [sp, #0x10]    ; Latency: 4
0000000100003190	fmla.4s	v18, v9, v3[0]          ; Latency: 2
0000000100003194	fmla.4s	v27, v8, v3[0]          ; Latency: 2
0000000100003198	ldr	q13, [x24, #0x1000]     ; Latency: 4
000000010000319c	fmla.4s	v22, v14, v13[0]        ; Latency: 2
00000001000031a0	fmla.4s	v23, v9, v13[0]         ; Latency: 2
00000001000031a4	fmla.4s	v29, v8, v13[0]         ; Latency: 2
00000001000031a8	ldr	q15, [x24, #0x1400]     ; Latency: 4
00000001000031ac	fmla.4s	v28, v14, v15[0]        ; Latency: 2
00000001000031b0	ldr	q5, [x23, #0x430]       ; Latency: 4
00000001000031b4	fmla.4s	v30, v9, v15[0]         ; Latency: 2
00000001000031b8	fmla.4s	v31, v8, v15[0]         ; Latency: 2
00000001000031bc	ldr	q8, [x23, #0x420]       ; Latency: 4
00000001000031c0	ldr	q9, [x23, #0x400]       ; Latency: 4
00000001000031c4	ldr	q4, [sp, #0x60]         ; Latency: 4
00000001000031c8	fmla.4s	v4, v0, v11[0]          ; Latency: 2
00000001000031cc	fmla.4s	v7, v0, v10[0]          ; Latency: 2
00000001000031d0	fmla.4s	v19, v0, v2[0]          ; Latency: 2
00000001000031d4	fmla.4s	v24, v0, v3[0]          ; Latency: 2
00000001000031d8	fmla.4s	v26, v0, v13[0]         ; Latency: 2
00000001000031dc	fmla.4s	v21, v0, v15[0]         ; Latency: 2
00000001000031e0	ldr	q0, [x23, #0x410]       ; Latency: 4
00000001000031e4	fmla.4s	v4, v0, v11[1]          ; Latency: 2
00000001000031e8	str	q4, [sp, #0x60]         ; Latency: 4
00000001000031ec	fmla.4s	v7, v0, v10[1]          ; Latency: 2
00000001000031f0	fmla.4s	v19, v0, v2[1]          ; Latency: 2
00000001000031f4	fmla.4s	v24, v0, v3[1]          ; Latency: 2
00000001000031f8	fmla.4s	v26, v0, v13[1]         ; Latency: 2
00000001000031fc	fmla.4s	v21, v0, v15[1]         ; Latency: 2
0000000100003200	ldr	q0, [x23, #0x810]       ; Latency: 4
0000000100003204	str	q0, [sp, #0x40]         ; Latency: 4
0000000100003208	fmla.4s	v6, v9, v11[1]          ; Latency: 2
000000010000320c	mov.16b	v17, v6
0000000100003210	fmla.4s	v20, v9, v10[1]         ; Latency: 2
0000000100003214	fmla.4s	v25, v9, v2[1]          ; Latency: 2
0000000100003218	fmla.4s	v27, v9, v3[1]          ; Latency: 2
000000010000321c	fmla.4s	v29, v9, v13[1]         ; Latency: 2
0000000100003220	fmla.4s	v31, v9, v15[1]         ; Latency: 2
0000000100003224	ldr	q9, [x23, #0x800]       ; Latency: 4
0000000100003228	mov.16b	v4, v1
000000010000322c	fmla.4s	v4, v8, v11[1]          ; Latency: 2
0000000100003230	fmla.4s	v12, v8, v10[1]         ; Latency: 2
0000000100003234	mov.16b	v14, v12
0000000100003238	fmla.4s	v16, v8, v2[1]          ; Latency: 2
000000010000323c	fmla.4s	v18, v8, v3[1]          ; Latency: 2
0000000100003240	fmla.4s	v23, v8, v13[1]         ; Latency: 2
0000000100003244	fmla.4s	v30, v8, v15[1]         ; Latency: 2
0000000100003248	ldr	q8, [x23, #0x820]       ; Latency: 4
000000010000324c	ldr	q1, [sp, #0x70]         ; Latency: 4
0000000100003250	mov.16b	v0, v5
0000000100003254	fmla.4s	v1, v5, v11[1]          ; Latency: 2
0000000100003258	ldr	q12, [sp, #0x50]        ; Latency: 4
000000010000325c	fmla.4s	v12, v5, v10[1]         ; Latency: 2
0000000100003260	ldp	q6, q5, [sp, #0x10]     ; Latency: 4
0000000100003264	fmla.4s	v5, v0, v2[1]           ; Latency: 2
0000000100003268	fmla.4s	v6, v0, v3[1]           ; Latency: 2
000000010000326c	fmla.4s	v22, v0, v13[1]         ; Latency: 2
0000000100003270	fmla.4s	v28, v0, v15[1]         ; Latency: 2
0000000100003274	ldr	q0, [x23, #0x830]       ; Latency: 4
0000000100003278	fmla.4s	v1, v0, v11[2]          ; Latency: 2
000000010000327c	str	q1, [sp, #0x70]         ; Latency: 4
0000000100003280	fmla.4s	v12, v0, v10[2]         ; Latency: 2
0000000100003284	str	q12, [sp, #0x50]        ; Latency: 4
0000000100003288	fmla.4s	v5, v0, v2[2]           ; Latency: 2
000000010000328c	fmla.4s	v6, v0, v3[2]           ; Latency: 2
0000000100003290	stp	q6, q5, [sp, #0x10]     ; Latency: 4
0000000100003294	fmla.4s	v22, v0, v13[2]         ; Latency: 2
0000000100003298	fmla.4s	v28, v0, v15[2]         ; Latency: 2
000000010000329c	ldr	q6, [x23, #0xc30]       ; Latency: 4
00000001000032a0	mov.16b	v1, v11
00000001000032a4	fmla.4s	v4, v8, v11[2]          ; Latency: 2
00000001000032a8	mov.16b	v12, v14
00000001000032ac	fmla.4s	v12, v8, v10[2]         ; Latency: 2
00000001000032b0	fmla.4s	v16, v8, v2[2]          ; Latency: 2
00000001000032b4	fmla.4s	v18, v8, v3[2]          ; Latency: 2
00000001000032b8	fmla.4s	v23, v8, v13[2]         ; Latency: 2
00000001000032bc	fmla.4s	v30, v8, v15[2]         ; Latency: 2
00000001000032c0	ldr	q0, [x23, #0xc20]       ; Latency: 4
00000001000032c4	fmla.4s	v17, v9, v11[2]         ; Latency: 2
00000001000032c8	fmla.4s	v20, v9, v10[2]         ; Latency: 2
00000001000032cc	fmla.4s	v25, v9, v2[2]          ; Latency: 2
00000001000032d0	fmla.4s	v27, v9, v3[2]          ; Latency: 2
00000001000032d4	fmla.4s	v29, v9, v13[2]         ; Latency: 2
00000001000032d8	fmla.4s	v31, v9, v15[2]         ; Latency: 2
00000001000032dc	ldr	q11, [x23, #0xc00]      ; Latency: 4
00000001000032e0	ldp	q5, q14, [sp, #0x60]    ; Latency: 4
00000001000032e4	ldr	q9, [sp, #0x40]         ; Latency: 4
00000001000032e8	fmla.4s	v5, v9, v1[2]           ; Latency: 2
00000001000032ec	mov.16b	v8, v1
00000001000032f0	fmla.4s	v7, v9, v10[2]          ; Latency: 2
00000001000032f4	fmla.4s	v19, v9, v2[2]          ; Latency: 2
00000001000032f8	fmla.4s	v24, v9, v3[2]          ; Latency: 2
00000001000032fc	fmla.4s	v26, v9, v13[2]         ; Latency: 2
0000000100003300	fmla.4s	v21, v9, v15[2]         ; Latency: 2
0000000100003304	ldr	q1, [x23, #0xc10]       ; Latency: 4
0000000100003308	fmla.4s	v5, v1, v8[3]           ; Latency: 2
000000010000330c	fmla.4s	v17, v11, v8[3]         ; Latency: 2
0000000100003310	fmla.4s	v4, v0, v8[3]           ; Latency: 2
0000000100003314	fmla.4s	v14, v6, v8[3]          ; Latency: 2
0000000100003318	ldr	q8, [x24, #0x10]        ; Latency: 4
000000010000331c	fmla.4s	v7, v1, v10[3]          ; Latency: 2
0000000100003320	fmla.4s	v20, v11, v10[3]        ; Latency: 2
0000000100003324	fmla.4s	v12, v0, v10[3]         ; Latency: 2
0000000100003328	stp	q4, q12, [sp, #0x30]    ; Latency: 4
000000010000332c	ldr	q4, [sp, #0x50]         ; Latency: 4
0000000100003330	mov.16b	v12, v6
0000000100003334	fmla.4s	v4, v6, v10[3]          ; Latency: 2
0000000100003338	stp	q4, q5, [sp, #0x50]     ; Latency: 4
000000010000333c	ldr	q9, [x24, #0x410]       ; Latency: 4
0000000100003340	fmla.4s	v19, v1, v2[3]          ; Latency: 2
0000000100003344	fmla.4s	v25, v11, v2[3]         ; Latency: 2
0000000100003348	fmla.4s	v16, v0, v2[3]          ; Latency: 2
000000010000334c	ldr	q5, [sp, #0x20]         ; Latency: 4
0000000100003350	fmla.4s	v5, v6, v2[3]           ; Latency: 2
0000000100003354	ldr	q4, [x24, #0x810]       ; Latency: 4
0000000100003358	fmla.4s	v24, v1, v3[3]          ; Latency: 2
000000010000335c	fmla.4s	v27, v11, v3[3]         ; Latency: 2
0000000100003360	fmla.4s	v18, v0, v3[3]          ; Latency: 2
0000000100003364	ldr	q6, [sp, #0x10]         ; Latency: 4
0000000100003368	fmla.4s	v6, v12, v3[3]          ; Latency: 2
000000010000336c	mov.16b	v2, v12
0000000100003370	ldr	q12, [x24, #0xc10]      ; Latency: 4
0000000100003374	fmla.4s	v26, v1, v13[3]         ; Latency: 2
0000000100003378	fmla.4s	v29, v11, v13[3]        ; Latency: 2
000000010000337c	fmla.4s	v23, v0, v13[3]         ; Latency: 2
0000000100003380	fmla.4s	v22, v2, v13[3]         ; Latency: 2
0000000100003384	ldr	q13, [x24, #0x1010]     ; Latency: 4
0000000100003388	fmla.4s	v21, v1, v15[3]         ; Latency: 2
000000010000338c	ldr	q3, [x24, #0x1410]      ; Latency: 4
0000000100003390	fmla.4s	v31, v11, v15[3]        ; Latency: 2
0000000100003394	ldr	q1, [x23, #0x1010]      ; Latency: 4
0000000100003398	fmla.4s	v30, v0, v15[3]         ; Latency: 2
000000010000339c	ldr	q0, [x23, #0x1000]      ; Latency: 4
00000001000033a0	fmla.4s	v28, v2, v15[3]         ; Latency: 2
00000001000033a4	ldr	q10, [x23, #0x1020]     ; Latency: 4
00000001000033a8	ldr	q11, [x23, #0x1030]     ; Latency: 4
00000001000033ac	fmla.4s	v14, v11, v8[0]         ; Latency: 2
00000001000033b0	str	q14, [sp, #0x70]        ; Latency: 4
00000001000033b4	ldp	q2, q15, [sp, #0x40]    ; Latency: 4
00000001000033b8	fmla.4s	v15, v11, v9[0]         ; Latency: 2
00000001000033bc	str	q15, [sp, #0x50]        ; Latency: 4
00000001000033c0	fmla.4s	v5, v11, v4[0]          ; Latency: 2
00000001000033c4	fmla.4s	v6, v11, v12[0]         ; Latency: 2
00000001000033c8	str	q6, [sp, #0x10]         ; Latency: 4
00000001000033cc	fmla.4s	v22, v11, v13[0]        ; Latency: 2
00000001000033d0	fmla.4s	v28, v11, v3[0]         ; Latency: 2
00000001000033d4	ldr	q11, [x23, #0x1430]     ; Latency: 4
00000001000033d8	ldr	q15, [sp, #0x30]        ; Latency: 4
00000001000033dc	fmla.4s	v15, v10, v8[0]         ; Latency: 2
00000001000033e0	fmla.4s	v2, v10, v9[0]          ; Latency: 2
00000001000033e4	fmla.4s	v16, v10, v4[0]         ; Latency: 2
00000001000033e8	fmla.4s	v18, v10, v12[0]        ; Latency: 2
00000001000033ec	fmla.4s	v23, v10, v13[0]        ; Latency: 2
00000001000033f0	fmla.4s	v30, v10, v3[0]         ; Latency: 2
00000001000033f4	ldr	q10, [x23, #0x1420]     ; Latency: 4
00000001000033f8	fmla.4s	v17, v0, v8[0]          ; Latency: 2
00000001000033fc	fmla.4s	v20, v0, v9[0]          ; Latency: 2
0000000100003400	fmla.4s	v25, v0, v4[0]          ; Latency: 2
0000000100003404	fmla.4s	v27, v0, v12[0]         ; Latency: 2
0000000100003408	fmla.4s	v29, v0, v13[0]         ; Latency: 2
000000010000340c	fmla.4s	v31, v0, v3[0]          ; Latency: 2
0000000100003410	ldr	q0, [x23, #0x1400]      ; Latency: 4
0000000100003414	ldr	q14, [sp, #0x60]        ; Latency: 4
0000000100003418	fmla.4s	v14, v1, v8[0]          ; Latency: 2
000000010000341c	fmla.4s	v7, v1, v9[0]           ; Latency: 2
0000000100003420	fmla.4s	v19, v1, v4[0]          ; Latency: 2
0000000100003424	fmla.4s	v24, v1, v12[0]         ; Latency: 2
0000000100003428	fmla.4s	v26, v1, v13[0]         ; Latency: 2
000000010000342c	fmla.4s	v21, v1, v3[0]          ; Latency: 2
0000000100003430	ldr	q1, [x23, #0x1410]      ; Latency: 4
0000000100003434	fmla.4s	v14, v1, v8[1]          ; Latency: 2
0000000100003438	str	q14, [sp, #0x60]        ; Latency: 4
000000010000343c	fmla.4s	v7, v1, v9[1]           ; Latency: 2
0000000100003440	str	q7, [sp]                ; Latency: 4
0000000100003444	fmla.4s	v19, v1, v4[1]          ; Latency: 2
0000000100003448	fmla.4s	v24, v1, v12[1]         ; Latency: 2
000000010000344c	fmla.4s	v26, v1, v13[1]         ; Latency: 2
0000000100003450	fmla.4s	v21, v1, v3[1]          ; Latency: 2
0000000100003454	ldr	q1, [x23, #0x1810]      ; Latency: 4
0000000100003458	fmla.4s	v17, v0, v8[1]          ; Latency: 2
000000010000345c	mov.16b	v7, v17
0000000100003460	fmla.4s	v20, v0, v9[1]          ; Latency: 2
0000000100003464	fmla.4s	v25, v0, v4[1]          ; Latency: 2
0000000100003468	fmla.4s	v27, v0, v12[1]         ; Latency: 2
000000010000346c	fmla.4s	v29, v0, v13[1]         ; Latency: 2
0000000100003470	fmla.4s	v31, v0, v3[1]          ; Latency: 2
0000000100003474	ldr	q0, [x23, #0x1800]      ; Latency: 4
0000000100003478	fmla.4s	v15, v10, v8[1]         ; Latency: 2
000000010000347c	fmla.4s	v2, v10, v9[1]          ; Latency: 2
0000000100003480	stp	q15, q2, [sp, #0x30]    ; Latency: 4
0000000100003484	fmla.4s	v16, v10, v4[1]         ; Latency: 2
0000000100003488	fmla.4s	v18, v10, v12[1]        ; Latency: 2
000000010000348c	fmla.4s	v23, v10, v13[1]        ; Latency: 2
0000000100003490	fmla.4s	v30, v10, v3[1]         ; Latency: 2
0000000100003494	ldr	q10, [x23, #0x1820]     ; Latency: 4
0000000100003498	ldr	q15, [sp, #0x70]        ; Latency: 4
000000010000349c	fmla.4s	v15, v11, v8[1]         ; Latency: 2
00000001000034a0	ldr	q2, [sp, #0x50]         ; Latency: 4
00000001000034a4	fmla.4s	v2, v11, v9[1]          ; Latency: 2
00000001000034a8	fmla.4s	v5, v11, v4[1]          ; Latency: 2
00000001000034ac	ldr	q17, [sp, #0x10]        ; Latency: 4
00000001000034b0	fmla.4s	v17, v11, v12[1]        ; Latency: 2
00000001000034b4	fmla.4s	v22, v11, v13[1]        ; Latency: 2
00000001000034b8	fmla.4s	v28, v11, v3[1]         ; Latency: 2
00000001000034bc	ldr	q11, [x23, #0x1830]     ; Latency: 4
00000001000034c0	fmla.4s	v15, v11, v8[2]         ; Latency: 2
00000001000034c4	str	q15, [sp, #0x70]        ; Latency: 4
00000001000034c8	fmla.4s	v2, v11, v9[2]          ; Latency: 2
00000001000034cc	fmla.4s	v5, v11, v4[2]          ; Latency: 2
00000001000034d0	fmla.4s	v17, v11, v12[2]        ; Latency: 2
00000001000034d4	stp	q17, q13, [sp, #0x10]   ; Latency: 4
00000001000034d8	mov.16b	v14, v13
00000001000034dc	fmla.4s	v22, v11, v13[2]        ; Latency: 2
00000001000034e0	mov.16b	v15, v3
00000001000034e4	fmla.4s	v28, v11, v3[2]         ; Latency: 2
00000001000034e8	ldr	q11, [x23, #0x1c30]     ; Latency: 4
00000001000034ec	ldp	q13, q3, [sp, #0x30]    ; Latency: 4
00000001000034f0	fmla.4s	v13, v10, v8[2]         ; Latency: 2
00000001000034f4	fmla.4s	v3, v10, v9[2]          ; Latency: 2
00000001000034f8	mov.16b	v6, v16
00000001000034fc	fmla.4s	v6, v10, v4[2]          ; Latency: 2
0000000100003500	fmla.4s	v18, v10, v12[2]        ; Latency: 2
0000000100003504	fmla.4s	v23, v10, v14[2]        ; Latency: 2
0000000100003508	fmla.4s	v30, v10, v15[2]        ; Latency: 2
000000010000350c	mov.16b	v10, v15
0000000100003510	ldr	q15, [x23, #0x1c20]     ; Latency: 4
0000000100003514	mov.16b	v16, v7
0000000100003518	fmla.4s	v16, v0, v8[2]          ; Latency: 2
000000010000351c	fmla.4s	v20, v0, v9[2]          ; Latency: 2
0000000100003520	fmla.4s	v25, v0, v4[2]          ; Latency: 2
0000000100003524	fmla.4s	v27, v0, v12[2]         ; Latency: 2
0000000100003528	fmla.4s	v29, v0, v14[2]         ; Latency: 2
000000010000352c	fmla.4s	v31, v0, v10[2]         ; Latency: 2
0000000100003530	mov.16b	v7, v10
0000000100003534	ldr	q0, [x23, #0x1c00]      ; Latency: 4
0000000100003538	ldr	q10, [sp, #0x60]        ; Latency: 4
000000010000353c	fmla.4s	v10, v1, v8[2]          ; Latency: 2
0000000100003540	ldr	q17, [sp]               ; Latency: 4
0000000100003544	fmla.4s	v17, v1, v9[2]          ; Latency: 2
0000000100003548	fmla.4s	v19, v1, v4[2]          ; Latency: 2
000000010000354c	fmla.4s	v24, v1, v12[2]         ; Latency: 2
0000000100003550	fmla.4s	v26, v1, v14[2]         ; Latency: 2
0000000100003554	fmla.4s	v21, v1, v7[2]          ; Latency: 2
0000000100003558	ldr	q1, [x23, #0x1c10]      ; Latency: 4
000000010000355c	fmla.4s	v10, v1, v8[3]          ; Latency: 2
0000000100003560	str	q10, [sp, #0x60]        ; Latency: 4
0000000100003564	fmla.4s	v16, v0, v8[3]          ; Latency: 2
0000000100003568	fmla.4s	v13, v15, v8[3]         ; Latency: 2
000000010000356c	ldr	q10, [sp, #0x70]        ; Latency: 4
0000000100003570	fmla.4s	v10, v11, v8[3]         ; Latency: 2
0000000100003574	str	q10, [sp, #0x70]        ; Latency: 4
0000000100003578	ldr	q10, [x24, #0x20]       ; Latency: 4
000000010000357c	fmla.4s	v17, v1, v9[3]          ; Latency: 2
0000000100003580	str	q17, [sp]               ; Latency: 4
0000000100003584	fmla.4s	v20, v0, v9[3]          ; Latency: 2
0000000100003588	fmla.4s	v3, v15, v9[3]          ; Latency: 2
000000010000358c	stp	q13, q3, [sp, #0x30]    ; Latency: 4
0000000100003590	fmla.4s	v2, v11, v9[3]          ; Latency: 2
0000000100003594	ldr	q9, [x24, #0x420]       ; Latency: 4
0000000100003598	fmla.4s	v19, v1, v4[3]          ; Latency: 2
000000010000359c	fmla.4s	v25, v0, v4[3]          ; Latency: 2
00000001000035a0	fmla.4s	v6, v15, v4[3]          ; Latency: 2
00000001000035a4	fmla.4s	v5, v11, v4[3]          ; Latency: 2
00000001000035a8	ldr	q4, [x24, #0x820]       ; Latency: 4
00000001000035ac	fmla.4s	v24, v1, v12[3]         ; Latency: 2
00000001000035b0	fmla.4s	v27, v0, v12[3]         ; Latency: 2
00000001000035b4	fmla.4s	v18, v15, v12[3]        ; Latency: 2
00000001000035b8	ldp	q17, q3, [sp, #0x10]    ; Latency: 4
00000001000035bc	fmla.4s	v17, v11, v12[3]        ; Latency: 2
00000001000035c0	ldr	q12, [x24, #0xc20]      ; Latency: 4
00000001000035c4	fmla.4s	v26, v1, v3[3]          ; Latency: 2
00000001000035c8	fmla.4s	v29, v0, v3[3]          ; Latency: 2
00000001000035cc	fmla.4s	v23, v15, v3[3]         ; Latency: 2
00000001000035d0	fmla.4s	v22, v11, v3[3]         ; Latency: 2
00000001000035d4	ldr	q13, [x24, #0x1020]     ; Latency: 4
00000001000035d8	fmla.4s	v21, v1, v7[3]          ; Latency: 2
00000001000035dc	ldr	q3, [x24, #0x1420]      ; Latency: 4
00000001000035e0	fmla.4s	v31, v0, v7[3]          ; Latency: 2
00000001000035e4	ldr	q0, [x23, #0x2010]      ; Latency: 4
00000001000035e8	fmla.4s	v30, v15, v7[3]         ; Latency: 2
00000001000035ec	ldr	q1, [x23, #0x2000]      ; Latency: 4
00000001000035f0	fmla.4s	v28, v11, v7[3]         ; Latency: 2
00000001000035f4	ldr	q11, [x23, #0x2020]     ; Latency: 4
00000001000035f8	ldr	q15, [x23, #0x2030]     ; Latency: 4
00000001000035fc	ldr	q8, [sp, #0x70]         ; Latency: 4
0000000100003600	fmla.4s	v8, v15, v10[0]         ; Latency: 2
0000000100003604	str	q8, [sp, #0x70]         ; Latency: 4
0000000100003608	fmla.4s	v2, v15, v9[0]          ; Latency: 2
000000010000360c	str	q2, [sp, #0x50]         ; Latency: 4
0000000100003610	fmla.4s	v5, v15, v4[0]          ; Latency: 2
0000000100003614	str	q5, [sp, #0x20]         ; Latency: 4
0000000100003618	fmla.4s	v17, v15, v12[0]        ; Latency: 2
000000010000361c	fmla.4s	v22, v15, v13[0]        ; Latency: 2
0000000100003620	fmla.4s	v28, v15, v3[0]         ; Latency: 2
0000000100003624	ldr	q15, [x23, #0x2430]     ; Latency: 4
0000000100003628	ldp	q2, q5, [sp, #0x30]     ; Latency: 4
000000010000362c	fmla.4s	v2, v11, v10[0]         ; Latency: 2
0000000100003630	fmla.4s	v5, v11, v9[0]          ; Latency: 2
0000000100003634	str	q5, [sp, #0x40]         ; Latency: 4
0000000100003638	fmla.4s	v6, v11, v4[0]          ; Latency: 2
000000010000363c	fmla.4s	v18, v11, v12[0]        ; Latency: 2
0000000100003640	fmla.4s	v23, v11, v13[0]        ; Latency: 2
0000000100003644	fmla.4s	v30, v11, v3[0]         ; Latency: 2
0000000100003648	ldr	q11, [x23, #0x2420]     ; Latency: 4
000000010000364c	fmla.4s	v16, v1, v10[0]         ; Latency: 2
0000000100003650	fmla.4s	v20, v1, v9[0]          ; Latency: 2
0000000100003654	fmla.4s	v25, v1, v4[0]          ; Latency: 2
0000000100003658	fmla.4s	v27, v1, v12[0]         ; Latency: 2
000000010000365c	fmla.4s	v29, v1, v13[0]         ; Latency: 2
0000000100003660	fmla.4s	v31, v1, v3[0]          ; Latency: 2
0000000100003664	ldr	q1, [x23, #0x2400]      ; Latency: 4
0000000100003668	ldr	q14, [sp, #0x60]        ; Latency: 4
000000010000366c	fmla.4s	v14, v0, v10[0]         ; Latency: 2
0000000100003670	ldr	q7, [sp]                ; Latency: 4
0000000100003674	fmla.4s	v7, v0, v9[0]           ; Latency: 2
0000000100003678	fmla.4s	v19, v0, v4[0]          ; Latency: 2
000000010000367c	mov.16b	v5, v4
0000000100003680	fmla.4s	v24, v0, v12[0]         ; Latency: 2
0000000100003684	fmla.4s	v26, v0, v13[0]         ; Latency: 2
0000000100003688	fmla.4s	v21, v0, v3[0]          ; Latency: 2
000000010000368c	ldr	q0, [x23, #0x2410]      ; Latency: 4
0000000100003690	fmla.4s	v14, v0, v10[1]         ; Latency: 2
0000000100003694	str	q14, [sp, #0x60]        ; Latency: 4
0000000100003698	fmla.4s	v7, v0, v9[1]           ; Latency: 2
000000010000369c	fmla.4s	v19, v0, v4[1]          ; Latency: 2
00000001000036a0	fmla.4s	v24, v0, v12[1]         ; Latency: 2
00000001000036a4	fmla.4s	v26, v0, v13[1]         ; Latency: 2
00000001000036a8	fmla.4s	v21, v0, v3[1]          ; Latency: 2
00000001000036ac	ldr	q8, [x23, #0x2810]      ; Latency: 4
00000001000036b0	fmla.4s	v16, v1, v10[1]         ; Latency: 2
00000001000036b4	fmla.4s	v20, v1, v9[1]          ; Latency: 2
00000001000036b8	fmla.4s	v25, v1, v4[1]          ; Latency: 2
00000001000036bc	fmla.4s	v27, v1, v12[1]         ; Latency: 2
00000001000036c0	fmla.4s	v29, v1, v13[1]         ; Latency: 2
00000001000036c4	fmla.4s	v31, v1, v3[1]          ; Latency: 2
00000001000036c8	ldr	q4, [x23, #0x2800]      ; Latency: 4
00000001000036cc	fmla.4s	v2, v11, v10[1]         ; Latency: 2
00000001000036d0	mov.16b	v14, v2
00000001000036d4	ldp	q0, q1, [sp, #0x40]     ; Latency: 4
00000001000036d8	fmla.4s	v0, v11, v9[1]          ; Latency: 2
00000001000036dc	str	q0, [sp, #0x40]         ; Latency: 4
00000001000036e0	fmla.4s	v6, v11, v5[1]          ; Latency: 2
00000001000036e4	fmla.4s	v18, v11, v12[1]        ; Latency: 2
00000001000036e8	fmla.4s	v23, v11, v13[1]        ; Latency: 2
00000001000036ec	fmla.4s	v30, v11, v3[1]         ; Latency: 2
00000001000036f0	ldr	q11, [x23, #0x2820]     ; Latency: 4
00000001000036f4	ldr	q0, [sp, #0x70]         ; Latency: 4
00000001000036f8	fmla.4s	v0, v15, v10[1]         ; Latency: 2
00000001000036fc	fmla.4s	v1, v15, v9[1]          ; Latency: 2
0000000100003700	ldr	q2, [sp, #0x20]         ; Latency: 4
0000000100003704	fmla.4s	v2, v15, v5[1]          ; Latency: 2
0000000100003708	fmla.4s	v17, v15, v12[1]        ; Latency: 2
000000010000370c	fmla.4s	v22, v15, v13[1]        ; Latency: 2
0000000100003710	fmla.4s	v28, v15, v3[1]         ; Latency: 2
0000000100003714	ldr	q15, [x23, #0x2830]     ; Latency: 4
0000000100003718	fmla.4s	v0, v15, v10[2]         ; Latency: 2
000000010000371c	str	q0, [sp, #0x70]         ; Latency: 4
0000000100003720	fmla.4s	v1, v15, v9[2]          ; Latency: 2
0000000100003724	str	q1, [sp, #0x50]         ; Latency: 4
0000000100003728	fmla.4s	v2, v15, v5[2]          ; Latency: 2
000000010000372c	str	q2, [sp, #0x20]         ; Latency: 4
0000000100003730	fmla.4s	v17, v15, v12[2]        ; Latency: 2
0000000100003734	fmla.4s	v22, v15, v13[2]        ; Latency: 2
0000000100003738	mov.16b	v0, v3
000000010000373c	fmla.4s	v28, v15, v3[2]         ; Latency: 2
0000000100003740	ldr	q15, [x23, #0x2c30]     ; Latency: 4
0000000100003744	mov.16b	v2, v14
0000000100003748	fmla.4s	v2, v11, v10[2]         ; Latency: 2
000000010000374c	ldr	q3, [sp, #0x40]         ; Latency: 4
0000000100003750	fmla.4s	v3, v11, v9[2]          ; Latency: 2
0000000100003754	fmla.4s	v6, v11, v5[2]          ; Latency: 2
0000000100003758	fmla.4s	v18, v11, v12[2]        ; Latency: 2
000000010000375c	fmla.4s	v23, v11, v13[2]        ; Latency: 2
0000000100003760	fmla.4s	v30, v11, v0[2]         ; Latency: 2
0000000100003764	ldr	q11, [x23, #0x2c20]     ; Latency: 4
0000000100003768	fmla.4s	v16, v4, v10[2]         ; Latency: 2
000000010000376c	fmla.4s	v20, v4, v9[2]          ; Latency: 2
0000000100003770	fmla.4s	v25, v4, v5[2]          ; Latency: 2
0000000100003774	fmla.4s	v27, v4, v12[2]         ; Latency: 2
0000000100003778	fmla.4s	v29, v4, v13[2]         ; Latency: 2
000000010000377c	fmla.4s	v31, v4, v0[2]          ; Latency: 2
0000000100003780	ldr	q1, [x23, #0x2c00]      ; Latency: 4
0000000100003784	ldr	q14, [sp, #0x60]        ; Latency: 4
0000000100003788	fmla.4s	v14, v8, v10[2]         ; Latency: 2
000000010000378c	fmla.4s	v7, v8, v9[2]           ; Latency: 2
0000000100003790	fmla.4s	v19, v8, v5[2]          ; Latency: 2
0000000100003794	fmla.4s	v24, v8, v12[2]         ; Latency: 2
0000000100003798	fmla.4s	v26, v8, v13[2]         ; Latency: 2
000000010000379c	fmla.4s	v21, v8, v0[2]          ; Latency: 2
00000001000037a0	mov.16b	v8, v0
00000001000037a4	ldr	q0, [x23, #0x2c10]      ; Latency: 4
00000001000037a8	fmla.4s	v14, v0, v10[3]         ; Latency: 2
00000001000037ac	fmla.4s	v16, v1, v10[3]         ; Latency: 2
00000001000037b0	fmla.4s	v2, v11, v10[3]         ; Latency: 2
00000001000037b4	str	q2, [sp, #0x30]         ; Latency: 4
00000001000037b8	ldr	q2, [sp, #0x70]         ; Latency: 4
00000001000037bc	fmla.4s	v2, v15, v10[3]         ; Latency: 2
00000001000037c0	str	q2, [sp, #0x70]         ; Latency: 4
00000001000037c4	ldr	q10, [x24, #0x30]       ; Latency: 4
00000001000037c8	fmla.4s	v7, v0, v9[3]           ; Latency: 2
00000001000037cc	fmla.4s	v20, v1, v9[3]          ; Latency: 2
00000001000037d0	mov.16b	v2, v3
00000001000037d4	fmla.4s	v2, v11, v9[3]          ; Latency: 2
00000001000037d8	str	q2, [sp, #0x40]         ; Latency: 4
00000001000037dc	ldr	q2, [sp, #0x50]         ; Latency: 4
00000001000037e0	fmla.4s	v2, v15, v9[3]          ; Latency: 2
00000001000037e4	ldr	q9, [x24, #0x430]       ; Latency: 4
00000001000037e8	mov.16b	v3, v5
00000001000037ec	fmla.4s	v19, v0, v5[3]          ; Latency: 2
00000001000037f0	fmla.4s	v25, v1, v5[3]          ; Latency: 2
00000001000037f4	fmla.4s	v6, v11, v5[3]          ; Latency: 2
00000001000037f8	ldr	q5, [sp, #0x20]         ; Latency: 4
00000001000037fc	fmla.4s	v5, v15, v3[3]          ; Latency: 2
0000000100003800	ldr	q4, [x24, #0x830]       ; Latency: 4
0000000100003804	fmla.4s	v24, v0, v12[3]         ; Latency: 2
0000000100003808	fmla.4s	v27, v1, v12[3]         ; Latency: 2
000000010000380c	fmla.4s	v18, v11, v12[3]        ; Latency: 2
0000000100003810	fmla.4s	v17, v15, v12[3]        ; Latency: 2
0000000100003814	ldr	q12, [x24, #0xc30]      ; Latency: 4
0000000100003818	fmla.4s	v26, v0, v13[3]         ; Latency: 2
000000010000381c	fmla.4s	v29, v1, v13[3]         ; Latency: 2
0000000100003820	fmla.4s	v23, v11, v13[3]        ; Latency: 2
0000000100003824	fmla.4s	v22, v15, v13[3]        ; Latency: 2
0000000100003828	ldr	q13, [x24, #0x1030]     ; Latency: 4
000000010000382c	fmla.4s	v21, v0, v8[3]          ; Latency: 2
0000000100003830	ldr	q0, [x24, #0x1430]      ; Latency: 4
0000000100003834	fmla.4s	v31, v1, v8[3]          ; Latency: 2
0000000100003838	ldr	q1, [x23, #0x3010]      ; Latency: 4
000000010000383c	fmla.4s	v30, v11, v8[3]         ; Latency: 2
0000000100003840	ldr	q11, [x23, #0x3000]     ; Latency: 4
0000000100003844	fmla.4s	v28, v15, v8[3]         ; Latency: 2
0000000100003848	ldr	q8, [x23, #0x3020]      ; Latency: 4
000000010000384c	ldr	q15, [x23, #0x3030]     ; Latency: 4
0000000100003850	ldr	q3, [sp, #0x70]         ; Latency: 4
0000000100003854	fmla.4s	v3, v15, v10[0]         ; Latency: 2
0000000100003858	str	q3, [sp, #0x70]         ; Latency: 4
000000010000385c	fmla.4s	v2, v15, v9[0]          ; Latency: 2
0000000100003860	str	q2, [sp, #0x50]         ; Latency: 4
0000000100003864	fmla.4s	v5, v15, v4[0]          ; Latency: 2
0000000100003868	fmla.4s	v17, v15, v12[0]        ; Latency: 2
000000010000386c	fmla.4s	v22, v15, v13[0]        ; Latency: 2
0000000100003870	mov.16b	v3, v0
0000000100003874	fmla.4s	v28, v15, v0[0]         ; Latency: 2
0000000100003878	ldr	q15, [x23, #0x3430]     ; Latency: 4
000000010000387c	ldp	q2, q0, [sp, #0x30]     ; Latency: 4
0000000100003880	fmla.4s	v2, v8, v10[0]          ; Latency: 2
0000000100003884	fmla.4s	v0, v8, v9[0]           ; Latency: 2
0000000100003888	fmla.4s	v6, v8, v4[0]           ; Latency: 2
000000010000388c	fmla.4s	v18, v8, v12[0]         ; Latency: 2
0000000100003890	fmla.4s	v23, v8, v13[0]         ; Latency: 2
0000000100003894	fmla.4s	v30, v8, v3[0]          ; Latency: 2
0000000100003898	ldr	q8, [x23, #0x3420]      ; Latency: 4
000000010000389c	fmla.4s	v16, v11, v10[0]        ; Latency: 2
00000001000038a0	fmla.4s	v20, v11, v9[0]         ; Latency: 2
00000001000038a4	fmla.4s	v25, v11, v4[0]         ; Latency: 2
00000001000038a8	fmla.4s	v27, v11, v12[0]        ; Latency: 2
00000001000038ac	fmla.4s	v29, v11, v13[0]        ; Latency: 2
00000001000038b0	fmla.4s	v31, v11, v3[0]         ; Latency: 2
00000001000038b4	ldr	q11, [x23, #0x3400]     ; Latency: 4
00000001000038b8	fmla.4s	v14, v1, v10[0]         ; Latency: 2
00000001000038bc	fmla.4s	v7, v1, v9[0]           ; Latency: 2
00000001000038c0	fmla.4s	v19, v1, v4[0]          ; Latency: 2
00000001000038c4	fmla.4s	v24, v1, v12[0]         ; Latency: 2
00000001000038c8	fmla.4s	v26, v1, v13[0]         ; Latency: 2
00000001000038cc	fmla.4s	v21, v1, v3[0]          ; Latency: 2
00000001000038d0	ldr	q1, [x23, #0x3410]      ; Latency: 4
00000001000038d4	fmla.4s	v14, v1, v10[1]         ; Latency: 2
00000001000038d8	str	q14, [sp, #0x60]        ; Latency: 4
00000001000038dc	fmla.4s	v7, v1, v9[1]           ; Latency: 2
00000001000038e0	fmla.4s	v19, v1, v4[1]          ; Latency: 2
00000001000038e4	fmla.4s	v24, v1, v12[1]         ; Latency: 2
00000001000038e8	fmla.4s	v26, v1, v13[1]         ; Latency: 2
00000001000038ec	mov.16b	v14, v3
00000001000038f0	fmla.4s	v21, v1, v3[1]          ; Latency: 2
00000001000038f4	ldr	q1, [x23, #0x3810]      ; Latency: 4
00000001000038f8	fmla.4s	v16, v11, v10[1]        ; Latency: 2
00000001000038fc	fmla.4s	v20, v11, v9[1]         ; Latency: 2
0000000100003900	fmla.4s	v25, v11, v4[1]         ; Latency: 2
0000000100003904	fmla.4s	v27, v11, v12[1]        ; Latency: 2
0000000100003908	fmla.4s	v29, v11, v13[1]        ; Latency: 2
000000010000390c	fmla.4s	v31, v11, v3[1]         ; Latency: 2
0000000100003910	ldr	q11, [x23, #0x3800]     ; Latency: 4
0000000100003914	fmla.4s	v2, v8, v10[1]          ; Latency: 2
0000000100003918	fmla.4s	v0, v8, v9[1]           ; Latency: 2
000000010000391c	stp	q2, q0, [sp, #0x30]     ; Latency: 4
0000000100003920	fmla.4s	v6, v8, v4[1]           ; Latency: 2
0000000100003924	fmla.4s	v18, v8, v12[1]         ; Latency: 2
0000000100003928	fmla.4s	v23, v8, v13[1]         ; Latency: 2
000000010000392c	fmla.4s	v30, v8, v3[1]          ; Latency: 2
0000000100003930	ldr	q8, [x23, #0x3820]      ; Latency: 4
0000000100003934	ldr	q3, [sp, #0x70]         ; Latency: 4
0000000100003938	fmla.4s	v3, v15, v10[1]         ; Latency: 2
000000010000393c	ldr	q2, [sp, #0x50]         ; Latency: 4
0000000100003940	fmla.4s	v2, v15, v9[1]          ; Latency: 2
0000000100003944	mov.16b	v0, v5
0000000100003948	fmla.4s	v0, v15, v4[1]          ; Latency: 2
000000010000394c	fmla.4s	v17, v15, v12[1]        ; Latency: 2
0000000100003950	fmla.4s	v22, v15, v13[1]        ; Latency: 2
0000000100003954	fmla.4s	v28, v15, v14[1]        ; Latency: 2
0000000100003958	ldr	q15, [x23, #0x3830]     ; Latency: 4
000000010000395c	fmla.4s	v3, v15, v10[2]         ; Latency: 2
0000000100003960	str	q3, [sp, #0x70]         ; Latency: 4
0000000100003964	fmla.4s	v2, v15, v9[2]          ; Latency: 2
0000000100003968	fmla.4s	v0, v15, v4[2]          ; Latency: 2
000000010000396c	str	q0, [sp, #0x20]         ; Latency: 4
0000000100003970	fmla.4s	v17, v15, v12[2]        ; Latency: 2
0000000100003974	fmla.4s	v22, v15, v13[2]        ; Latency: 2
0000000100003978	fmla.4s	v28, v15, v14[2]        ; Latency: 2
000000010000397c	ldr	q15, [x23, #0x3c30]     ; Latency: 4
0000000100003980	ldp	q0, q3, [sp, #0x30]     ; Latency: 4
0000000100003984	fmla.4s	v0, v8, v10[2]          ; Latency: 2
0000000100003988	fmla.4s	v3, v8, v9[2]           ; Latency: 2
000000010000398c	fmla.4s	v6, v8, v4[2]           ; Latency: 2
0000000100003990	fmla.4s	v18, v8, v12[2]         ; Latency: 2
0000000100003994	fmla.4s	v23, v8, v13[2]         ; Latency: 2
0000000100003998	fmla.4s	v30, v8, v14[2]         ; Latency: 2
000000010000399c	ldr	q8, [x23, #0x3c20]      ; Latency: 4
00000001000039a0	fmla.4s	v16, v11, v10[2]        ; Latency: 2
00000001000039a4	fmla.4s	v20, v11, v9[2]         ; Latency: 2
00000001000039a8	fmla.4s	v25, v11, v4[2]         ; Latency: 2
00000001000039ac	fmla.4s	v27, v11, v12[2]        ; Latency: 2
00000001000039b0	fmla.4s	v29, v11, v13[2]        ; Latency: 2
00000001000039b4	fmla.4s	v31, v11, v14[2]        ; Latency: 2
00000001000039b8	ldr	q11, [x23, #0x3c00]     ; Latency: 4
00000001000039bc	ldr	q5, [sp, #0x60]         ; Latency: 4
00000001000039c0	fmla.4s	v5, v1, v10[2]          ; Latency: 2
00000001000039c4	fmla.4s	v7, v1, v9[2]           ; Latency: 2
00000001000039c8	fmla.4s	v19, v1, v4[2]          ; Latency: 2
00000001000039cc	fmla.4s	v24, v1, v12[2]         ; Latency: 2
00000001000039d0	fmla.4s	v26, v1, v13[2]         ; Latency: 2
00000001000039d4	fmla.4s	v21, v1, v14[2]         ; Latency: 2
00000001000039d8	ldr	q1, [x23, #0x3c10]      ; Latency: 4
00000001000039dc	fmla.4s	v5, v1, v10[3]          ; Latency: 2
00000001000039e0	str	q5, [sp, #0x60]         ; Latency: 4
00000001000039e4	fmla.4s	v16, v11, v10[3]        ; Latency: 2
00000001000039e8	fmla.4s	v0, v8, v10[3]          ; Latency: 2
00000001000039ec	str	q0, [sp, #0x30]         ; Latency: 4
00000001000039f0	ldr	q0, [sp, #0x70]         ; Latency: 4
00000001000039f4	fmla.4s	v0, v15, v10[3]         ; Latency: 2
00000001000039f8	mov.16b	v10, v0
00000001000039fc	fmla.4s	v7, v1, v9[3]           ; Latency: 2
0000000100003a00	fmla.4s	v20, v11, v9[3]         ; Latency: 2
0000000100003a04	fmla.4s	v3, v8, v9[3]           ; Latency: 2
0000000100003a08	fmla.4s	v2, v15, v9[3]          ; Latency: 2
0000000100003a0c	fmla.4s	v19, v1, v4[3]          ; Latency: 2
0000000100003a10	fmla.4s	v25, v11, v4[3]         ; Latency: 2
0000000100003a14	fmla.4s	v6, v8, v4[3]           ; Latency: 2
0000000100003a18	mov.16b	v5, v6
0000000100003a1c	ldr	q0, [sp, #0x20]         ; Latency: 4
0000000100003a20	fmla.4s	v0, v15, v4[3]          ; Latency: 2
0000000100003a24	mov.16b	v4, v0
0000000100003a28	fmla.4s	v24, v1, v12[3]         ; Latency: 2
0000000100003a2c	fmla.4s	v27, v11, v12[3]        ; Latency: 2
0000000100003a30	fmla.4s	v18, v8, v12[3]         ; Latency: 2
0000000100003a34	fmla.4s	v17, v15, v12[3]        ; Latency: 2
0000000100003a38	fmla.4s	v26, v1, v13[3]         ; Latency: 2
0000000100003a3c	fmla.4s	v29, v11, v13[3]        ; Latency: 2
0000000100003a40	fmla.4s	v23, v8, v13[3]         ; Latency: 2
0000000100003a44	fmla.4s	v22, v15, v13[3]        ; Latency: 2
0000000100003a48	fmla.4s	v21, v1, v14[3]         ; Latency: 2
0000000100003a4c	fmla.4s	v31, v11, v14[3]        ; Latency: 2
0000000100003a50	fmla.4s	v30, v8, v14[3]         ; Latency: 2
0000000100003a54	fmla.4s	v28, v15, v14[3]        ; Latency: 2
0000000100003a58	add	x22, x22, #0x10
0000000100003a5c	add	x24, x24, x15           ; Latency: 2
0000000100003a60	add	x23, x23, x0            ; Latency: 2
0000000100003a64	cmp	x22, #0xf0
0000000100003a68	b.lo	0x100003134
0000000100003a6c	ldr	q0, [sp, #0x60]         ; Latency: 4
0000000100003a70	stp	q16, q0, [x5]           ; Latency: 4
0000000100003a74	ldr	q0, [sp, #0x30]         ; Latency: 4
0000000100003a78	stp	q0, q10, [x5, #0x20]    ; Latency: 4
0000000100003a7c	stp	q20, q7, [x6]           ; Latency: 4
0000000100003a80	stp	q3, q2, [x6, #0x20]     ; Latency: 4
0000000100003a84	stp	q25, q19, [x7]          ; Latency: 4
0000000100003a88	stp	q5, q4, [x7, #0x20]     ; Latency: 4
0000000100003a8c	stp	q27, q24, [x19]         ; Latency: 4
0000000100003a90	stp	q18, q17, [x19, #0x20]  ; Latency: 4
0000000100003a94	stp	q29, q26, [x21]         ; Latency: 4
0000000100003a98	stp	q23, q22, [x21, #0x20]  ; Latency: 4
0000000100003a9c	stp	q30, q28, [x20, #0x20]  ; Latency: 4
0000000100003aa0	add	x5, x2, #0x10
0000000100003aa4	add	x4, x4, x17             ; Latency: 2
0000000100003aa8	stp	q31, q21, [x20]         ; Latency: 4
0000000100003aac	cmp	x2, #0xf0
0000000100003ab0	mov	x2, x5                  ; Latency: 2
0000000100003ab4	b.lo	0x1000030ac
0000000100003ab8	add	x2, x10, #0x6
0000000100003abc	add	x13, x13, x14           ; Latency: 2
0000000100003ac0	cmp	x10, #0xba
0000000100003ac4	mov	x10, x2                 ; Latency: 2
0000000100003ac8	b.lo	0x1000030a0
0000000100003acc	ldp	x20, x19, [sp, #0xe0]   ; Latency: 4
0000000100003ad0	ldp	x22, x21, [sp, #0xd0]   ; Latency: 4
0000000100003ad4	ldp	x24, x23, [sp, #0xc0]   ; Latency: 4
0000000100003ad8	ldp	d9, d8, [sp, #0xb0]     ; Latency: 4
0000000100003adc	ldp	d11, d10, [sp, #0xa0]   ; Latency: 4
0000000100003ae0	ldp	d13, d12, [sp, #0x90]   ; Latency: 4
0000000100003ae4	ldp	d15, d14, [sp, #0x80]   ; Latency: 4
0000000100003ae8	add	sp, sp, #0xf0
0000000100003aec	ret
_rtclock:
0000000100003af0	sub	sp, sp, #0x30
0000000100003af4	stp	x29, x30, [sp, #0x20]   ; Latency: 6
0000000100003af8	add	x29, sp, #0x20
0000000100003afc	add	x0, sp, #0x8
0000000100003b00	sub	x1, x29, #0x8
0000000100003b04	bl	0x100003e74 ; symbol stub for: _gettimeofday
0000000100003b08	cbz	w0, 0x100003b1c
0000000100003b0c	str	x0, [sp]                ; Latency: 4
0000000100003b10	adr	x0, #0x408 ; literal pool for: "Error return from gettimeofday: %d"
0000000100003b14	nop
0000000100003b18	bl	0x100003e8c ; symbol stub for: _printf
0000000100003b1c	ldr	d0, [sp, #0x8]          ; Latency: 4
0000000100003b20	scvtf	d0, d0                  ; Latency: 2
0000000100003b24	ldr	s1, [sp, #0x10]         ; Latency: 4
0000000100003b28	sshll.2d	v1, v1, #0x0    ; Latency: 2
0000000100003b2c	scvtf	d1, d1                  ; Latency: 2
0000000100003b30	nop
0000000100003b34	ldr	d2, 0x100003f10         ; Latency: 4
0000000100003b38	fmul	d1, d1, d2              ; Latency: 5
0000000100003b3c	fadd	d0, d1, d0              ; Latency: 5
0000000100003b40	ldp	x29, x30, [sp, #0x20]   ; Latency: 4
0000000100003b44	add	sp, sp, #0x30
0000000100003b48	ret
_init_matrix:
0000000100003b4c	stp	x26, x25, [sp, #-0x50]! ; Latency: 6
0000000100003b50	stp	x24, x23, [sp, #0x10]   ; Latency: 6
0000000100003b54	stp	x22, x21, [sp, #0x20]   ; Latency: 6
0000000100003b58	stp	x20, x19, [sp, #0x30]   ; Latency: 6
0000000100003b5c	stp	x29, x30, [sp, #0x40]   ; Latency: 6
0000000100003b60	add	x29, sp, #0x40
0000000100003b64	cmp	w2, #0x1
0000000100003b68	b.lt	0x100003bc0
0000000100003b6c	cmp	w1, #0x1
0000000100003b70	b.lt	0x100003bc0
0000000100003b74	mov	x19, x0                 ; Latency: 2
0000000100003b78	mov	x20, #0x0
0000000100003b7c	mov	w21, w1                 ; Latency: 2
0000000100003b80	mov	w22, w2                 ; Latency: 2
0000000100003b84	lsl	x23, x21, #2
0000000100003b88	mov	w24, #0x30000000
0000000100003b8c	mov	x25, x21                ; Latency: 2
0000000100003b90	mov	x26, x19                ; Latency: 2
0000000100003b94	bl	0x100003e98 ; symbol stub for: _rand
0000000100003b98	scvtf	s0, w0                  ; Latency: 10
0000000100003b9c	fmov	s1, w24                 ; Latency: 5
0000000100003ba0	fmul	s0, s0, s1              ; Latency: 4
0000000100003ba4	str	s0, [x26], #0x4         ; Latency: 4
0000000100003ba8	subs	x25, x25, #0x1
0000000100003bac	b.ne	0x100003b94
0000000100003bb0	add	x20, x20, #0x1
0000000100003bb4	add	x19, x19, x23           ; Latency: 2
0000000100003bb8	cmp	x20, x22                ; Latency: 2
0000000100003bbc	b.ne	0x100003b8c
0000000100003bc0	ldp	x29, x30, [sp, #0x40]   ; Latency: 4
0000000100003bc4	ldp	x20, x19, [sp, #0x30]   ; Latency: 4
0000000100003bc8	ldp	x22, x21, [sp, #0x20]   ; Latency: 4
0000000100003bcc	ldp	x24, x23, [sp, #0x10]   ; Latency: 4
0000000100003bd0	ldp	x26, x25, [sp], #0x50   ; Latency: 4
0000000100003bd4	ret
_main:
0000000100003bd8	sub	sp, sp, #0xf0
0000000100003bdc	stp	d9, d8, [sp, #0x80]     ; Latency: 6
0000000100003be0	stp	x28, x27, [sp, #0x90]   ; Latency: 6
0000000100003be4	stp	x26, x25, [sp, #0xa0]   ; Latency: 6
0000000100003be8	stp	x24, x23, [sp, #0xb0]   ; Latency: 6
0000000100003bec	stp	x22, x21, [sp, #0xc0]   ; Latency: 6
0000000100003bf0	stp	x20, x19, [sp, #0xd0]   ; Latency: 6
0000000100003bf4	stp	x29, x30, [sp, #0xe0]   ; Latency: 6
0000000100003bf8	add	x29, sp, #0xe0
0000000100003bfc	mov	w8, #0x64
0000000100003c00	mov	w9, #0x100
0000000100003c04	stp	x9, x8, [sp, #0x10]     ; Latency: 6
0000000100003c08	mov	w8, #0xc0
0000000100003c0c	stp	x8, x9, [sp]            ; Latency: 6
0000000100003c10	adr	x0, #0x32b ; literal pool for: "Benchmarking MLIR %d x %d x %d [%d times] \n"
0000000100003c14	nop
0000000100003c18	bl	0x100003e8c ; symbol stub for: _printf
0000000100003c1c	add	x0, sp, #0x68
0000000100003c20	sub	x1, x29, #0x68
0000000100003c24	bl	0x100003e74 ; symbol stub for: _gettimeofday
0000000100003c28	cbz	w0, 0x100003c3c
0000000100003c2c	str	x0, [sp]                ; Latency: 4
0000000100003c30	adr	x0, #0x2e8 ; literal pool for: "Error return from gettimeofday: %d"
0000000100003c34	nop
0000000100003c38	bl	0x100003e8c ; symbol stub for: _printf
0000000100003c3c	ldr	x22, [sp, #0x68]        ; Latency: 4
0000000100003c40	ldr	w23, [sp, #0x70]        ; Latency: 4
0000000100003c44	mov	w0, #0x30000
0000000100003c48	bl	0x100003e80 ; symbol stub for: _malloc
0000000100003c4c	mov	x19, x0                 ; Latency: 2
0000000100003c50	mov	w0, #0x40000
0000000100003c54	bl	0x100003e80 ; symbol stub for: _malloc
0000000100003c58	mov	x20, x0                 ; Latency: 2
0000000100003c5c	mov	w0, #0x30000
0000000100003c60	bl	0x100003e80 ; symbol stub for: _malloc
0000000100003c64	mov	x21, x0                 ; Latency: 2
0000000100003c68	mov	x24, #0x0
0000000100003c6c	mov	w25, #0x30000000
0000000100003c70	mov	x26, x19                ; Latency: 2
0000000100003c74	mov	x27, #0x0
0000000100003c78	bl	0x100003e98 ; symbol stub for: _rand
0000000100003c7c	scvtf	s0, w0                  ; Latency: 10
0000000100003c80	fmov	s1, w25                 ; Latency: 5
0000000100003c84	fmul	s0, s0, s1              ; Latency: 4
0000000100003c88	str	s0, [x26, x27]
0000000100003c8c	add	x27, x27, #0x4
0000000100003c90	cmp	x27, #0x300
0000000100003c94	b.ne	0x100003c78
0000000100003c98	add	x24, x24, #0x1
0000000100003c9c	add	x26, x26, #0x300
0000000100003ca0	cmp	x24, #0x100
0000000100003ca4	b.ne	0x100003c74
0000000100003ca8	mov	x24, #0x0
0000000100003cac	mov	w25, #0x30000000
0000000100003cb0	mov	x26, x20                ; Latency: 2
0000000100003cb4	mov	x27, #0x0
0000000100003cb8	bl	0x100003e98 ; symbol stub for: _rand
0000000100003cbc	scvtf	s0, w0                  ; Latency: 10
0000000100003cc0	fmov	s1, w25                 ; Latency: 5
0000000100003cc4	fmul	s0, s0, s1              ; Latency: 4
0000000100003cc8	str	s0, [x26, x27]
0000000100003ccc	add	x27, x27, #0x4
0000000100003cd0	cmp	x27, #0x400
0000000100003cd4	b.ne	0x100003cb8
0000000100003cd8	add	x24, x24, #0x1
0000000100003cdc	add	x26, x26, #0x400
0000000100003ce0	cmp	x24, #0x100
0000000100003ce4	b.ne	0x100003cb4
0000000100003ce8	mov	x24, #0x0
0000000100003cec	mov	w25, #0x30000000
0000000100003cf0	mov	x26, x21                ; Latency: 2
0000000100003cf4	mov	x27, #0x0
0000000100003cf8	bl	0x100003e98 ; symbol stub for: _rand
0000000100003cfc	scvtf	s0, w0                  ; Latency: 10
0000000100003d00	fmov	s1, w25                 ; Latency: 5
0000000100003d04	fmul	s0, s0, s1              ; Latency: 4
0000000100003d08	str	s0, [x26, x27]
0000000100003d0c	add	x27, x27, #0x4
0000000100003d10	cmp	x27, #0x300
0000000100003d14	b.ne	0x100003cf8
0000000100003d18	add	x24, x24, #0x1
0000000100003d1c	add	x26, x26, #0x300
0000000100003d20	cmp	x24, #0x100
0000000100003d24	b.ne	0x100003cf4
0000000100003d28	mov	w24, #0x64
0000000100003d2c	mov	w25, #0xc0
0000000100003d30	mov	w26, #0x1
0000000100003d34	mov	w27, #0x100
0000000100003d38	stp	x26, x25, [sp, #0x58]   ; Latency: 6
0000000100003d3c	stp	x25, x27, [sp, #0x48]   ; Latency: 6
0000000100003d40	stp	x21, xzr, [sp, #0x38]   ; Latency: 6
0000000100003d44	stp	x27, x21, [sp, #0x28]   ; Latency: 6
0000000100003d48	stp	x27, x26, [sp, #0x18]   ; Latency: 6
0000000100003d4c	stp	xzr, x27, [sp, #0x8]    ; Latency: 6
0000000100003d50	str	x20, [sp]               ; Latency: 4
0000000100003d54	mov	x0, x19                 ; Latency: 2
0000000100003d58	mov	x1, x19                 ; Latency: 2
0000000100003d5c	mov	x2, #0x0
0000000100003d60	mov	w3, #0xc0
0000000100003d64	mov	w4, #0x100
0000000100003d68	mov	w5, #0x1
0000000100003d6c	mov	w6, #0xc0
0000000100003d70	mov	x7, x20                 ; Latency: 2
0000000100003d74	bl	_matmul
0000000100003d78	subs	w24, w24, #0x1
0000000100003d7c	b.ne	0x100003d38
0000000100003d80	add	x0, sp, #0x68
0000000100003d84	sub	x1, x29, #0x68
0000000100003d88	bl	0x100003e74 ; symbol stub for: _gettimeofday
0000000100003d8c	cbz	w0, 0x100003da0
0000000100003d90	str	x0, [sp]                ; Latency: 4
0000000100003d94	adr	x0, #0x184 ; literal pool for: "Error return from gettimeofday: %d"
0000000100003d98	nop
0000000100003d9c	bl	0x100003e8c ; symbol stub for: _printf
0000000100003da0	scvtf	d0, w23                 ; Latency: 10
0000000100003da4	nop
0000000100003da8	ldr	d1, 0x100003f10         ; Latency: 4
0000000100003dac	fmul	d0, d0, d1              ; Latency: 5
0000000100003db0	scvtf	d2, x22                 ; Latency: 10
0000000100003db4	fadd	d8, d0, d2              ; Latency: 5
0000000100003db8	ldr	d0, [sp, #0x68]         ; Latency: 4
0000000100003dbc	scvtf	d0, d0                  ; Latency: 2
0000000100003dc0	ldr	s2, [sp, #0x70]         ; Latency: 4
0000000100003dc4	sshll.2d	v2, v2, #0x0    ; Latency: 2
0000000100003dc8	scvtf	d2, d2                  ; Latency: 2
0000000100003dcc	fmul	d1, d2, d1              ; Latency: 5
0000000100003dd0	fadd	d9, d1, d0              ; Latency: 5
0000000100003dd4	adr	x0, #0x193 ; literal pool for: "matmul_192x256x256_mlir_perf.out"
0000000100003dd8	nop
0000000100003ddc	adr	x1, #0x1ac ; literal pool for: "w"
0000000100003de0	nop
0000000100003de4	bl	0x100003e5c ; symbol stub for: _fopen
0000000100003de8	mov	x19, x0                 ; Latency: 2
0000000100003dec	fsub	d0, d9, d8              ; Latency: 5
0000000100003df0	mov	x8, #0xc00000000000
0000000100003df4	movk	x8, #0x41e2, lsl #48
0000000100003df8	fmov	d1, x8                  ; Latency: 5
0000000100003dfc	fdiv	d0, d1, d0              ; Latency: 17
0000000100003e00	mov	x8, #0xcd6500000000
0000000100003e04	movk	x8, #0x41cd, lsl #48
0000000100003e08	fmov	d1, x8                  ; Latency: 5
0000000100003e0c	fdiv	d0, d0, d1              ; Latency: 17
0000000100003e10	str	d0, [sp]                ; Latency: 4
0000000100003e14	adr	x1, #0x176 ; literal pool for: "%0.2lf GFLOPS\n"
0000000100003e18	nop
0000000100003e1c	bl	0x100003e68 ; symbol stub for: _fprintf
0000000100003e20	mov	x0, x19                 ; Latency: 2
0000000100003e24	bl	0x100003e50 ; symbol stub for: _fclose
0000000100003e28	mov	w0, #0x0
0000000100003e2c	ldp	x29, x30, [sp, #0xe0]   ; Latency: 4
0000000100003e30	ldp	x20, x19, [sp, #0xd0]   ; Latency: 4
0000000100003e34	ldp	x22, x21, [sp, #0xc0]   ; Latency: 4
0000000100003e38	ldp	x24, x23, [sp, #0xb0]   ; Latency: 4
0000000100003e3c	ldp	x26, x25, [sp, #0xa0]   ; Latency: 4
0000000100003e40	ldp	x28, x27, [sp, #0x90]   ; Latency: 4
0000000100003e44	ldp	d9, d8, [sp, #0x80]     ; Latency: 4
0000000100003e48	add	sp, sp, #0xf0
0000000100003e4c	ret
