"""Fluxo multi-turno de qualificação, exercitando o checkpointing.

Duas mensagens na MESMA thread: a segunda deve enxergar o que a primeira
coletou (memória via checkpointer), provando a persistência entre mensagens
de WhatsApp.
"""

from __future__ import annotations


async def test_perfil_persiste_entre_mensagens(make_graph, new_thread_config, run_turn):
    graph = make_graph()
    config = new_thread_config("+5541999990001")

    # Turno 1: lead diz que quer um SUV até 140 mil.
    s1 = await run_turn(graph, "Quero um SUV até 140 mil pra família", config)
    lead1 = s1["lead_profile"]
    assert lead1.body_type == "suv"
    assert lead1.budget_max == 140000.0
    # Já deve ter buscado opções (tem critério) e respondido.
    assert s1["final_reply"]
    assert s1["retrieved_vehicles"]

    # Turno 2: lead só pergunta de financiamento, sem repetir o perfil.
    s2 = await run_turn(graph, "Como fica o financiamento do primeiro?", config)
    # O perfil do turno 1 sobreviveu (checkpointing).
    assert s2["lead_profile"].body_type == "suv"
    assert s2["lead_profile"].budget_max == 140000.0
    assert s2["financing_quote"] is not None


async def test_threads_diferentes_nao_compartilham_memoria(make_graph, new_thread_config, run_turn):
    graph = make_graph()
    c1 = new_thread_config("+5541000000001")
    c2 = new_thread_config("+5541000000002")

    await run_turn(graph, "Quero um sedan Toyota", c1)
    s2 = await run_turn(graph, "Oi, tudo bem?", c2)
    # Lead 2 é novo: não herda o interesse do lead 1.
    assert s2["lead_profile"].desired_make is None


async def test_veiculos_respeitam_orcamento(make_graph, new_thread_config, run_turn):
    graph = make_graph()
    config = new_thread_config()
    s = await run_turn(graph, "Me mostra SUVs até 120 mil", config)
    assert s["retrieved_vehicles"]
    assert all(v.price <= 120000.0 for v in s["retrieved_vehicles"])
    assert all(v.body_type == "suv" for v in s["retrieved_vehicles"])
