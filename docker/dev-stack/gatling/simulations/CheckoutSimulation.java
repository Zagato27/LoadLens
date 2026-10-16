package devstack;

import static io.gatling.javaapi.core.CoreDsl.*;
import static io.gatling.javaapi.http.HttpDsl.*;

import io.gatling.javaapi.core.*;
import io.gatling.javaapi.http.*;
import java.time.Duration;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;

/**
 * Dev-stack Gatling run for the lt-gatling-graphite-victoriametrics pack: a group, names with spaces,
 * a request that always fails (KO), an open and a closed scenario. Metrics go to VictoriaMetrics
 * through the Graphite writer (gatling.conf next to this file).
 */
public class CheckoutSimulation extends Simulation {

  static final Function<Session, String> ITEM = session -> "/items/" + ThreadLocalRandom.current().nextInt(1, 1000);

  HttpProtocolBuilder protocol = http.baseUrl("http://pyapp:8000");

  ScenarioBuilder buyers = scenario("Buyers")
      .group("Catalog").on(
          exec(http("Item page").get(ITEM))
              .exec(http("Item again").get(ITEM)))
      .exec(http("Checkout fails").get("/fail"));

  ScenarioBuilder browsers = scenario("Browsers")
      .forever().on(exec(http("Item page").get(ITEM)).pause(Duration.ofMillis(300)));

  // -DstepSeconds=40 shortens the run (the Run test page of LoadLens passes it in JAVA_OPTS).
  static final Duration STEP = Duration.ofSeconds(Long.getLong("stepSeconds", 120L));

  {
    setUp(
        // Open model, steps of 10 → 20 → 30 users per second, two minutes each by default.
        buyers.injectOpen(
            constantUsersPerSec(10).during(STEP),
            constantUsersPerSec(20).during(STEP),
            constantUsersPerSec(30).during(STEP)),
        browsers.injectClosed(constantConcurrentUsers(5).during(STEP.multipliedBy(3))))
        .maxDuration(STEP.multipliedBy(3).plusSeconds(10))
        .protocols(protocol);
  }
}
